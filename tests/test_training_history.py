from src.config import Config
from src.dqn import DQNAgent
from src.train import run_training


def test_training_loop_runs_with_recurrent_histories(tmp_path):
    cfg = Config()
    cfg.grid_width = 6
    cfg.grid_height = 6
    cfg.num_agents = 2
    cfg.num_apples = 4
    cfg.max_steps_per_episode = 5
    cfg.batch_size = 2
    cfg.min_buffer_size = 2
    cfg.learn_every = 1
    cfg.checkpoint_dir = str(tmp_path)
    cfg.checkpoint_every = 100
    cfg.log_every = 100

    run_training(cfg, num_episodes=1, render=False)

    assert (tmp_path / "dqn_final.pt").exists()


def test_training_loop_uses_batched_action_selection(monkeypatch, tmp_path):
    cfg = Config()
    cfg.grid_width = 6
    cfg.grid_height = 6
    cfg.num_agents = 3
    cfg.num_apples = 4
    cfg.max_steps_per_episode = 2
    cfg.min_buffer_size = 999
    cfg.checkpoint_dir = str(tmp_path)
    cfg.checkpoint_every = 100
    cfg.log_every = 100

    batch_calls = []

    def fail_single_action(*args, **kwargs):
        raise AssertionError("training should use batched action selection")

    def fake_act_batch(self, state_batch):
        batch_calls.append(state_batch.shape[0])
        return [0 for _ in range(state_batch.shape[0])]

    monkeypatch.setattr(DQNAgent, "act", fail_single_action)
    monkeypatch.setattr(DQNAgent, "act_batch", fake_act_batch, raising=False)

    run_training(cfg, num_episodes=1, render=False)

    assert batch_calls
    assert all(call_size > 0 for call_size in batch_calls)
