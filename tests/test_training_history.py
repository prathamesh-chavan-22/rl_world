from src.config import Config
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
