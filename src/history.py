from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from src.config import Config


NO_ACTION_INDEX = 4


@dataclass
class SequenceHistory:
    """Fixed-length per-agent sequence of observation + previous-action frames."""

    frames: List[np.ndarray]
    cfg: Config

    @classmethod
    def from_initial_obs(cls, obs: np.ndarray, cfg: Config) -> "SequenceHistory":
        frame = cls._encode_frame(obs, NO_ACTION_INDEX, cfg)
        return cls([frame.copy() for _ in range(cfg.history_len)], cfg)

    @classmethod
    def from_frames(cls, frames: Iterable[np.ndarray], cfg: Config) -> "SequenceHistory":
        copied = [np.asarray(frame, dtype=np.float32).copy() for frame in frames]
        if len(copied) != cfg.history_len:
            raise ValueError(f"Expected {cfg.history_len} frames, got {len(copied)}")
        for frame in copied:
            if frame.shape != (cfg.rnn_input_dim,):
                raise ValueError(f"Expected frame shape {(cfg.rnn_input_dim,)}, got {frame.shape}")
        return cls(copied, cfg)

    def append(self, obs: np.ndarray, prev_action: int) -> "SequenceHistory":
        self.frames = self.frames[1:] + [self._encode_frame(obs, prev_action, self.cfg)]
        return self

    def copy(self) -> "SequenceHistory":
        return SequenceHistory.from_frames(self.frames, self.cfg)

    def as_array(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.float32, copy=True)

    @staticmethod
    def _encode_frame(obs: np.ndarray, prev_action: int, cfg: Config) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (cfg.obs_dim,):
            raise ValueError(f"Expected observation shape {(cfg.obs_dim,)}, got {obs.shape}")
        if prev_action < 0 or prev_action >= cfg.prev_action_dim:
            raise ValueError(f"Invalid previous action index {prev_action}")

        action_one_hot = np.zeros(cfg.prev_action_dim, dtype=np.float32)
        action_one_hot[prev_action] = 1.0
        return np.concatenate([obs, action_one_hot]).astype(np.float32)
