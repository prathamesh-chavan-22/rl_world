import torch.nn as nn

from src.config import Config
from src.dqn import RecurrentQNetwork


def test_default_config_uses_larger_model_tuning_profile():
    cfg = Config()

    assert cfg.history_len == 64
    assert cfg.reward_eat == 0.25
    assert cfg.reward_step_alive == 0.001
    assert cfg.reward_starvation == -0.25
    assert cfg.rnn_hidden_size == 192
    assert cfg.rnn_layers == 3
    assert cfg.q_head_hidden_sizes == (128, 64)


def test_recurrent_q_head_matches_deeper_architecture():
    cfg = Config()
    net = RecurrentQNetwork(cfg)
    linear_layers = [layer for layer in net.head if isinstance(layer, nn.Linear)]

    assert net.lstm.hidden_size == 192
    assert net.lstm.num_layers == 3
    assert [layer.in_features for layer in linear_layers] == [192, 128, 64]
    assert [layer.out_features for layer in linear_layers] == [128, 64, cfg.action_dim]
