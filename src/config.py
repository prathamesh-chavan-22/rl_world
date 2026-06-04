from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------ world
    grid_width: int = 20
    grid_height: int = 20
    num_agents: int = 6          # initial population (3M + 3F)
    num_apples: int = 15         # apples on the grid at any time (respawn)
    apple_respawn_prob: float = 0.02  # per empty cell per step
    max_steps_per_episode: int = 2000

    # ------------------------------------------------------------------ agents
    vision_range: int = 4        # cells the agent can see ahead
    hunger_max: float = 1.0      # full hunger
    hunger_decay: float = 0.005  # hunger lost per step
    hunger_eat_gain: float = 0.6 # hunger gained per apple eaten (capped at max)

    # ------------------------------------------------------------------ rewards  (all in [-1, 1])
    reward_eat: float = 0.25
    reward_step_alive: float = 0.001
    reward_starvation: float = -0.25
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    # ------------------------------------------------------------------ DQN
    obs_dim: int = 17            # 4 cells * 4 types one-hot + 1 hunger scalar
    action_dim: int = 4
    hidden_sizes: tuple = (64, 64)
    history_len: int = 64       # recurrent context length
    rnn_hidden_size: int = 192
    rnn_layers: int = 3
    q_head_hidden_sizes: tuple = (128, 64)

    # ------------------------------------------------------------------ training
    device: str = "auto"        # auto: cuda -> mps -> cpu
    precision: str = "auto"     # auto: cuda bf16/fp16, mps fp16, cpu fp32
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    target_sync_every: int = 200  # steps between hard target-net updates
    learn_every: int = 4          # learn every N environment steps
    min_buffer_size: int = 500    # don't learn until buffer has this many entries

    # regularisation
    weight_decay: float = 1e-4   # L2 via Adam weight_decay
    l1_lambda: float = 1e-5      # L1 penalty coefficient added per learn() call

    # ------------------------------------------------------------------ epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000  # linear decay over this many env steps

    # ------------------------------------------------------------------ checkpointing / logging
    checkpoint_every: int = 100  # save model every N episodes
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10          # print console metrics every N episodes
    use_tqdm: bool = True

    # ------------------------------------------------------------------ render
    cell_size: int = 32          # pixels per grid cell
    fps: int = 30
    render_hunger_bar: bool = True

    @property
    def prev_action_dim(self) -> int:
        """One-hot action size plus a final no-action/start marker."""
        return self.action_dim + 1

    @property
    def rnn_input_dim(self) -> int:
        """Per-timestep recurrent input: observation + previous action."""
        return self.obs_dim + self.prev_action_dim
