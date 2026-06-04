import numpy as np
import torch

from src.config import Config
from src.dqn import DQNAgent, RecurrentQNetwork


def test_recurrent_q_network_returns_one_q_value_per_action():
    cfg = Config()
    net = RecurrentQNetwork(cfg)
    batch = torch.zeros((3, cfg.history_len, cfg.rnn_input_dim), dtype=torch.float32)

    q_values = net(batch)

    assert q_values.shape == (3, cfg.action_dim)


def test_dqn_can_learn_from_sequence_transition():
    cfg = Config()
    cfg.batch_size = 2
    cfg.min_buffer_size = 2
    cfg.learn_every = 1
    cfg.buffer_size = 10
    dqn = DQNAgent(cfg, device="cpu")

    seq = np.zeros((cfg.history_len, cfg.rnn_input_dim), dtype=np.float32)
    next_seq = np.ones((cfg.history_len, cfg.rnn_input_dim), dtype=np.float32)

    dqn.push(seq, 0, 0.5, next_seq, False)
    dqn.push(next_seq, 1, -0.5, seq, True)

    loss = dqn.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_returns_one_greedy_action_per_sequence_in_batch():
    cfg = Config()
    dqn = DQNAgent(cfg, device="cpu")
    state_batch = np.zeros((4, cfg.history_len, cfg.rnn_input_dim), dtype=np.float32)

    actions = dqn.act_greedy_batch(state_batch)

    assert len(actions) == 4
    assert all(isinstance(action, int) for action in actions)
    assert all(0 <= action < cfg.action_dim for action in actions)


def test_dqn_batched_epsilon_actions_are_valid_for_each_sequence():
    cfg = Config()
    dqn = DQNAgent(cfg, device="cpu")
    dqn.epsilon = 1.0
    state_batch = np.zeros((6, cfg.history_len, cfg.rnn_input_dim), dtype=np.float32)

    actions = dqn.act_batch(state_batch)

    assert len(actions) == 6
    assert all(isinstance(action, int) for action in actions)
    assert all(0 <= action < cfg.action_dim for action in actions)
