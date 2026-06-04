import numpy as np

from src.config import Config
from src.history import NO_ACTION_INDEX, SequenceHistory


def test_initial_history_repeats_first_observation_with_no_action_marker():
    cfg = Config()
    obs = np.linspace(0.0, 1.0, cfg.obs_dim, dtype=np.float32)

    history = SequenceHistory.from_initial_obs(obs, cfg)
    seq = history.as_array()

    assert seq.shape == (cfg.history_len, cfg.rnn_input_dim)
    np.testing.assert_allclose(seq[:, : cfg.obs_dim], np.tile(obs, (cfg.history_len, 1)))
    assert np.all(seq[:, cfg.obs_dim + NO_ACTION_INDEX] == 1.0)
    assert np.all(seq[:, cfg.obs_dim : cfg.obs_dim + cfg.action_dim] == 0.0)


def test_append_shifts_history_and_encodes_previous_action():
    cfg = Config()
    first_obs = np.zeros(cfg.obs_dim, dtype=np.float32)
    next_obs = np.ones(cfg.obs_dim, dtype=np.float32)

    history = SequenceHistory.from_initial_obs(first_obs, cfg)
    updated = history.copy().append(next_obs, prev_action=2)
    seq = updated.as_array()

    np.testing.assert_allclose(seq[-1, : cfg.obs_dim], next_obs)
    assert seq[-1, cfg.obs_dim + 2] == 1.0
    assert seq[-1, cfg.obs_dim + NO_ACTION_INDEX] == 0.0
    np.testing.assert_allclose(seq[0, : cfg.obs_dim], first_obs)
