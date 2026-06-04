# RL Box World

A 2D discrete grid simulation where box agents (male/female) learn via a shared Deep Q-Network (DQN) to find and eat apples, managing their hunger while keeping their civilization alive as long as possible.

## Concept

- Agents are colored boxes on a grid; blue = male, pink = female.
- Each agent has a **hunger** stat that decays every step. Eating an apple refills it.
- An agent **dies** (starves) when hunger reaches 0.
- The episode ends when **all agents have starved** (civilization collapse) or the step cap is reached.
- All agents share a single recurrent Q-network policy — learning is pooled across the whole population.
- Agents can only see **4 cells directly ahead** of them (not behind, not sideways).
- Each decision uses the agent's last **16 timesteps** so it can learn short-term movement patterns.
- Gender is a first-class field, structured for future reproduction mechanics.

## Network Input (`16 x 22`)

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

At episode start, the first observation is repeated 16 times with the no-action marker. The network therefore receives a tensor shaped `16 x 22`, processes it with an LSTM, and outputs 4 Q-values.

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
| Eat apple      | +1.0   |
| Survive a step | +0.01  |
| Starvation     | -1.0   |

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

# Watch a trained model (no training, render only)
python main.py --render --no-train --checkpoint checkpoints/dqn_ep500.pt
```

Note: checkpoints created before the recurrent/LSTM update are not compatible with the current model architecture. Train a fresh checkpoint after this change.

## Project Structure

```
rl_world/
├── main.py              # Entry point
├── requirements.txt
├── checkpoints/         # Saved model weights (auto-created)
├── src/
│   ├── config.py        # All hyperparameters and world settings
│   ├── agent.py         # Agent entity (pos, orientation, gender, hunger)
│   ├── history.py       # Per-agent 16-step recurrent input history
│   ├── world.py         # GridWorld environment
│   ├── dqn.py           # RecurrentQNetwork, ReplayBuffer, DQNAgent
│   ├── render.py        # Pygame renderer
│   └── train.py         # Training loop
```

## Hyperparameters

All tunable in `src/config.py` — grid size, agent count, apple count, hunger decay, history length, LSTM size/layers, epsilon schedule, L1/L2 regularization, learning rate, etc.
