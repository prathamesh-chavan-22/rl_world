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
