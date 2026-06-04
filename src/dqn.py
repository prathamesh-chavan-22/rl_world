from __future__ import annotations

import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class RecurrentQNetwork(nn.Module):
    """LSTM Q-network over fixed-length observation/action history."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.rnn_input_dim,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.rnn_hidden_size, cfg.post_rnn_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.post_rnn_hidden_size, cfg.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        return self.head(final_hidden)


class ReplayBuffer:
    """Uniform replay buffer for fixed-length sequence transitions."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state_seq: np.ndarray,
        action: int,
        reward: float,
        next_state_seq: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state_seq, action, reward, next_state_seq, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self._buf, batch_size)
        state_seq, acts, rews, next_state_seq, dones = zip(*batch)
        return (
            torch.tensor(np.array(state_seq), dtype=torch.float32),
            torch.tensor(acts, dtype=torch.long),
            torch.tensor(rews, dtype=torch.float32),
            torch.tensor(np.array(next_state_seq), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    """Shared-policy recurrent DQN for all box agents."""

    def __init__(self, cfg: Config, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_net = RecurrentQNetwork(cfg).to(self.device)
        self.target_net = RecurrentQNetwork(cfg).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.total_steps = 0
        self.learn_steps = 0
        self.epsilon = cfg.epsilon_start

    def act(self, state_seq: np.ndarray) -> int:
        """Epsilon-greedy action selection from a 16-step sequence."""
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        return self.act_greedy(state_seq)

    def act_greedy(self, state_seq: np.ndarray) -> int:
        """Pure greedy action selection for evaluation."""
        seq_t = torch.tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(seq_t)
        return int(q_vals.argmax(dim=1).item())

    def step_epsilon(self) -> None:
        """Decay epsilon linearly."""
        self.total_steps += 1
        frac = min(self.total_steps / self.cfg.epsilon_decay_steps, 1.0)
        self.epsilon = self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def push(
        self,
        state_seq: np.ndarray,
        action: int,
        reward: float,
        next_state_seq: np.ndarray,
        done: bool,
    ) -> None:
        """Store a clipped-reward sequence transition in replay."""
        reward = float(np.clip(reward, self.cfg.reward_clip_min, self.cfg.reward_clip_max))
        self.buffer.push(state_seq, action, reward, next_state_seq, done)

    def learn(self) -> float | None:
        """Run one recurrent DQN update if enough replay data exists."""
        if len(self.buffer) < self.cfg.min_buffer_size:
            return None
        if self.learn_steps % self.cfg.learn_every != 0:
            self.learn_steps += 1
            return None

        state_seq, actions, rewards, next_state_seq, dones = self.buffer.sample(
            self.cfg.batch_size
        )
        state_seq = state_seq.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state_seq = next_state_seq.to(self.device)
        dones = dones.to(self.device)

        q_vals = self.online_net(state_seq).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state_seq).max(dim=1).values
            td_target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        td_loss = F.smooth_l1_loss(q_vals, td_target)
        l1_penalty = sum(p.abs().sum() for p in self.online_net.parameters())
        loss = td_loss + self.cfg.l1_lambda * l1_penalty

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "architecture": "recurrent_lstm_dqn",
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "learn_steps": self.learn_steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.learn_steps = checkpoint["learn_steps"]
        self.epsilon = checkpoint["epsilon"]
        self.target_net.eval()
