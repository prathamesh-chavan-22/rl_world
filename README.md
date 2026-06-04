# RL Box World

A 2D discrete grid simulation where box agents (male/female) learn via a shared Deep Q-Network (DQN) to find and eat apples, managing their hunger while keeping their civilization alive as long as possible.

## Concept

- Agents are colored boxes on a grid; blue = male, pink = female.
- Each agent has a **hunger** stat that decays every step. Eating an apple refills it.
- An agent **dies** (starves) when hunger reaches 0.
- The episode ends when **all agents have starved** (civilization collapse) or the step cap is reached.
- All agents share a single recurrent Q-network policy — learning is pooled across the whole population.
- Action selection is batched: all alive agents' histories are stacked into one model call per environment step for better MPS/CUDA utilization.
- Agents can only see **4 cells directly ahead** of them (not behind, not sideways).
- Each decision uses the agent's last **64 timesteps** so it can learn short-term movement patterns.
- Gender is a first-class field, structured for future reproduction mechanics.

## Network Input (`64 x 22`)

The world still emits one 17-number observation per agent:

```
[cell_0_empty, cell_0_apple, cell_0_wall, cell_0_agent,   # cell 1 ahead (4 dims)
 cell_1_empty, cell_1_apple, cell_1_wall, cell_1_agent,   # cell 2 ahead
 cell_2_empty, cell_2_apple, cell_2_wall, cell_2_agent,   # cell 3 ahead
 cell_3_empty, cell_3_apple, cell_3_wall, cell_3_agent,   # cell 4 ahead
 hunger_normalized]                                        # scalar in [0,1]
```

For the recurrent DQN, each timestep becomes:

```
17 observation values + 5 previous-action values = 22 features
```

The previous-action vector is one-hot:

| Index | Meaning |
|-------|---------|
| 0     | Previous action was move forward |
| 1     | Previous action was move backward |
| 2     | Previous action was turn left |
| 3     | Previous action was turn right |
| 4     | No previous action / episode start |

At episode start, the first observation is repeated 64 times with the no-action marker. The network therefore receives a tensor shaped `64 x 22`, processes it with a 3-layer LSTM (`hidden_size=192`), then maps the final hidden state through a `192 -> 128 -> 64 -> 4` Q-head.

## Action Space (4 discrete)

| ID | Action      |
|----|-------------|
| 0  | Move forward |
| 1  | Move backward |
| 2  | Turn left   |
| 3  | Turn right  |

## Rewards (bounded to [-1, 1])

| Event          | Reward |
|----------------|--------|
| Eat apple      | +0.25  |
| Survive a step | +0.001 |
| Starvation     | -0.25  |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train with live pygame render (watch agents learn)
python main.py --render

# Train headless (fast, no window)
python main.py

# Specify number of episodes
python main.py --episodes 2000

# Auto-pick the best PyTorch backend: CUDA -> Apple MPS -> CPU
python main.py --device auto

# Force Apple GPU through PyTorch's MPS backend (local Mac)
python main.py --device mps

# Force CUDA on an NVIDIA machine later
python main.py --device cuda

# Auto-pick precision from the chosen backend
python main.py --device auto --precision auto

# Force fp32 if mixed precision is unstable on a backend
python main.py --device mps --precision fp32

# Force CUDA bf16/fp16 when supported
python main.py --device cuda --precision bf16

# Watch a trained model (no training, render only)
python main.py --render --no-train --checkpoint checkpoints/dqn_ep500.pt
```

Note: checkpoints created before the recurrent/LSTM update are not compatible with the current model architecture. Train a fresh checkpoint after this change.

## GPU Backend

This project stays on PyTorch so it can run on Apple GPU now and CUDA later without rewriting the RL code. The default `--device auto` selection order is:

1. `cuda` if `torch.cuda.is_available()`
2. `mps` if `torch.backends.mps.is_available()`
3. `cpu` fallback

On Apple Silicon this uses PyTorch's Metal/MPS backend, not a full rewrite to Apple's separate MLX framework. That keeps future CUDA runs straightforward.

Precision is also configurable with `--precision auto|fp32|fp16|bf16`. The default `auto` policy is:

1. `cuda`: use `bf16` when `torch.cuda.is_bf16_supported()`, otherwise `fp16` with `GradScaler`
2. `mps`: use `fp16` autocast
3. `cpu`: keep `fp32`

Model weights and checkpoints stay in fp32. Mixed precision is applied with PyTorch autocast during model forward/loss computation, which is safer than storing replay observations or model weights in half precision.

## Project Structure

```
rl_world/
├── main.py              # Entry point
├── requirements.txt
├── checkpoints/         # Saved model weights (auto-created)
├── src/
│   ├── config.py        # All hyperparameters and world settings
│   ├── agent.py         # Agent entity (pos, orientation, gender, hunger)
│   ├── history.py       # Per-agent 64-step recurrent input history
│   ├── world.py         # GridWorld environment
│   ├── dqn.py           # RecurrentQNetwork, ReplayBuffer, DQNAgent
│   ├── render.py        # Pygame renderer
│   └── train.py         # Training loop
```

## Hyperparameters

All tunable in `src/config.py` — grid size, agent count, apple count, hunger decay, history length, LSTM size/layers, device backend, precision mode, epsilon schedule, L1/L2 regularization, learning rate, etc.
