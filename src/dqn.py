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


# ---------------------------------------------------------------------------
# Neural network

class QNetwork(nn.Module):
    """Small MLP: obs_dim -> hidden -> hidden -> action_dim."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.obs_dim
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer

class ReplayBuffer:
    """Uniform random experience replay buffer, shared across all agents."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self._buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(acts,               dtype=torch.long),
            torch.tensor(rews,               dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN agent (wraps network + buffer + training)

class DQNAgent:
    """Shared-policy DQN: one network for all box agents.

    Learning objective (each update):
        loss = SmoothL1(TD_target - Q(s,a))
               + l1_lambda * sum(|w|)     <- explicit L1 regularisation
        Optimiser uses weight_decay for L2 regularisation.
    """

    def __init__(self, cfg: Config, device: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.online_net = QNetwork(cfg).to(self.device)
        self.target_net = QNetwork(cfg).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,   # L2
        )

        self.buffer = ReplayBuffer(cfg.buffer_size)

        # counters
        self.total_steps: int = 0    # env steps (shared across agents in episode)
        self.learn_steps: int = 0    # gradient updates

        # linear epsilon decay
        self.epsilon: float = cfg.epsilon_start

    # ------------------------------------------------------------------
    # action selection

    def act(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def act_greedy(self, obs: np.ndarray) -> int:
        """Pure greedy (no exploration) — use for evaluation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # step / learn

    def step_epsilon(self) -> None:
        """Decay epsilon linearly."""
        self.total_steps += 1
        frac = min(self.total_steps / self.cfg.epsilon_decay_steps, 1.0)
        self.epsilon = self.cfg.epsilon_start + frac * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer (reward already clipped)."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def learn(self) -> float | None:
        """One gradient update. Returns scalar loss or None if skipped."""
        if len(self.buffer) < self.cfg.min_buffer_size:
            return None
        if self.learn_steps % self.cfg.learn_every != 0:
            self.learn_steps += 1
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)
        obs      = obs.to(self.device)
        actions  = actions.to(self.device)
        rewards  = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones    = dones.to(self.device)

        # current Q-values for chosen actions
        q_vals = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target using target network
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            td_target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        # Huber / smooth-L1 loss (less sensitive to outliers than MSE)
        td_loss = F.smooth_l1_loss(q_vals, td_target)

        # explicit L1 weight penalty
        l1_penalty = sum(p.abs().sum() for p in self.online_net.parameters())
        loss = td_loss + self.cfg.l1_lambda * l1_penalty

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1

        # hard sync target network
        if self.learn_steps % self.cfg.target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # persistence

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
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
        self.learn_steps  = checkpoint["learn_steps"]
        self.epsilon      = checkpoint["epsilon"]
        self.target_net.eval()
