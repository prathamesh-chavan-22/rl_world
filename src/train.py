from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from src.config import Config
from src.dqn import DQNAgent
from src.history import SequenceHistory
from src.world import GridWorld


def run_training(
    cfg: Config,
    num_episodes: int,
    render: bool = False,
    no_train: bool = False,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Main training loop.

    Shared-policy multi-agent:
      - All alive agents act using the same QNetwork.
      - All transitions go into the same replay buffer.
      - One learn() call happens per env step (if ready).

    Episode ends when all agents starve or step cap is reached.
    Headline metric: steps survived per episode (civilization longevity).
    """
    world = GridWorld(cfg)
    dqn   = DQNAgent(cfg)
    print(f"[train] Device: {dqn.device}")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[train] Loading checkpoint: {checkpoint_path}")
        dqn.load(checkpoint_path)

    renderer = None
    if render:
        from src.render import Renderer
        renderer = Renderer(cfg)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # rolling windows for console log
    win_steps:  Deque[int]   = deque(maxlen=cfg.log_every)
    win_apples: Deque[int]   = deque(maxlen=cfg.log_every)
    win_deaths: Deque[int]   = deque(maxlen=cfg.log_every)
    win_reward: Deque[float] = deque(maxlen=cfg.log_every)
    win_loss:   Deque[float] = deque(maxlen=cfg.log_every)

    total_env_steps = 0
    recent_loss: Optional[float] = None
    recent_reward: Optional[float] = None

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs_list = world.reset()
        # obs_list is indexed in world.agents order
        agents = world.agents

        history_map = {
            a.id: SequenceHistory.from_initial_obs(obs_list[i], cfg)
            for i, a in enumerate(agents)
        }

        ep_rewards: List[float] = []

        while not world.is_done():
            # --- collect actions for all alive agents ---
            actions = {}
            alive_agents = world.alive_agents()
            if alive_agents:
                state_batch = np.stack(
                    [history_map[agent.id].as_array() for agent in alive_agents],
                    axis=0,
                )
                if no_train:
                    batch_actions = dqn.act_greedy_batch(state_batch)
                else:
                    batch_actions = dqn.act_batch(state_batch)
                actions = {
                    agent.id: action
                    for agent, action in zip(alive_agents, batch_actions)
                }

            # --- step world ---
            next_obs_list, rewards, dones = world.step_with_eat_rewards(actions)

            # --- store transitions + learn ---
            for i, agent in enumerate(agents):
                if not (agent.id in actions):
                    # already dead before this step
                    continue
                state_seq = history_map[agent.id].as_array()
                action   = actions[agent.id]
                reward   = rewards[i]
                next_obs = next_obs_list[i]
                done     = dones[i]
                next_history = history_map[agent.id].copy().append(next_obs, action)
                next_state_seq = next_history.as_array()

                if not no_train:
                    dqn.push(state_seq, action, reward, next_state_seq, done)

                history_map[agent.id] = next_history
                ep_rewards.append(reward)

            if not no_train:
                loss = dqn.learn()
                if loss is not None:
                    recent_loss = loss
                    win_loss.append(loss)
                dqn.step_epsilon()

            total_env_steps += 1

            # --- optional render ---
            if renderer is not None:
                keep_open = renderer.draw(
                    world,
                    episode,
                    total_env_steps,
                    dqn.epsilon,
                    recent_loss,
                    sum(ep_rewards) / max(len(ep_rewards), 1),
                )
                if not keep_open:
                    print("[train] Window closed — stopping.")
                    if renderer:
                        renderer.close()
                    return

        # --- episode done ---
        summary = world.civilization_summary()
        ep_mean_reward = sum(ep_rewards) / max(len(ep_rewards), 1)

        win_steps.append(summary["steps_survived"])
        win_apples.append(summary["total_apples_eaten"])
        win_deaths.append(summary["total_deaths"])
        win_reward.append(ep_mean_reward)
        if recent_loss is not None:
            win_loss.append(recent_loss)

        recent_reward = ep_mean_reward

        # --- console metrics ---
        if episode % cfg.log_every == 0:
            avg_steps  = sum(win_steps)  / len(win_steps)
            avg_apples = sum(win_apples) / len(win_apples)
            avg_deaths = sum(win_deaths) / len(win_deaths)
            avg_reward = sum(win_reward) / len(win_reward)
            avg_loss   = sum(win_loss)   / len(win_loss) if win_loss else 0.0
            elapsed    = time.time() - start_time

            print(
                f"Ep {episode:5d} | "
                f"survive {avg_steps:6.1f} steps | "
                f"apples {avg_apples:5.1f} | "
                f"deaths {avg_deaths:4.1f} | "
                f"rew {avg_reward:+.3f} | "
                f"loss {avg_loss:.4f} | "
                f"ε {dqn.epsilon:.3f} | "
                f"{elapsed:.0f}s"
            )

        # --- checkpoint ---
        if not no_train and episode % cfg.checkpoint_every == 0:
            ckpt = os.path.join(cfg.checkpoint_dir, f"dqn_ep{episode:05d}.pt")
            dqn.save(ckpt)
            print(f"  [ckpt] saved → {ckpt}")

    # --- final checkpoint ---
    if not no_train:
        final_ckpt = os.path.join(cfg.checkpoint_dir, "dqn_final.pt")
        dqn.save(final_ckpt)
        print(f"[train] Training complete. Final checkpoint: {final_ckpt}")

    if renderer:
        renderer.close()
