from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from src.agent import Action, Agent, Gender, Orientation
from src.config import Config

# cell type indices for one-hot encoding
CELL_EMPTY = 0
CELL_APPLE = 1
CELL_WALL  = 2
CELL_AGENT = 3
NUM_CELL_TYPES = 4


class GridWorld:
    """Discrete 2-D grid environment for the RL box agents."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.W = config.grid_width
        self.H = config.grid_height

        self.agents: List[Agent] = []
        # apple_grid[y][x] = True if an apple is on that cell
        self.apple_grid: np.ndarray = np.zeros((self.H, self.W), dtype=bool)
        self.step_count: int = 0

        # episode-level metrics
        self.total_apples_eaten: int = 0
        self.total_deaths: int = 0
        self.population_history: List[int] = []

    # ------------------------------------------------------------------
    # reset / init

    def reset(self) -> List[np.ndarray]:
        """Reset world and return initial observations for all agents."""
        Agent._id_counter = 0
        self.apple_grid[:] = False
        self.step_count = 0
        self.total_apples_eaten = 0
        self.total_deaths = 0
        self.population_history = []

        self._spawn_agents()
        self._spawn_apples(self.cfg.num_apples)
        return [self._observe(a) for a in self.agents]

    def _spawn_agents(self) -> None:
        self.agents = []
        n = self.cfg.num_agents
        half = n // 2
        occupied: set = set()

        for i in range(n):
            gender = Gender.MALE if i < half else Gender.FEMALE
            pos = self._random_empty_pos(occupied)
            occupied.add(pos)
            orientation = random.choice(list(Orientation))
            self.agents.append(Agent(pos[0], pos[1], orientation, gender, self.cfg))

    def _spawn_apples(self, count: int) -> None:
        placed = 0
        agent_positions = {(a.x, a.y) for a in self.agents}
        attempts = 0
        while placed < count and attempts < count * 20:
            x = random.randint(0, self.W - 1)
            y = random.randint(0, self.H - 1)
            if not self.apple_grid[y, x] and (x, y) not in agent_positions:
                self.apple_grid[y, x] = True
                placed += 1
            attempts += 1

    def _random_empty_pos(self, occupied: set) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.W - 1)
            y = random.randint(0, self.H - 1)
            if (x, y) not in occupied:
                return (x, y)

    # ------------------------------------------------------------------
    # step

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """Advance the world by one step.

        Args:
            actions: mapping agent.id -> action int for each alive agent.

        Returns:
            obs list, reward list, done list — one entry per agent
            (alive or freshly dead) in self.agents order.
        """
        self.step_count += 1
        observations: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []

        alive_before = {a.id for a in self.agents if a.alive}

        # 1. apply actions for all alive agents
        for agent in self.agents:
            if not agent.alive:
                continue
            action = Action(actions.get(agent.id, Action.FORWARD))
            agent.apply_action(action, self.W, self.H)

        # 2. resolve eating (agent on apple cell)
        for agent in self.agents:
            if not agent.alive:
                continue
            if self.apple_grid[agent.y, agent.x]:
                self.apple_grid[agent.y, agent.x] = False
                agent.eat()
                self.total_apples_eaten += 1

        # 3. decay hunger + check starvation
        for agent in self.agents:
            if not agent.alive:
                continue
            agent.decay_hunger()
            agent.check_starvation()

        # 4. respawn apples
        self._respawn_apples()

        # 5. build outputs
        for agent in self.agents:
            just_died = (agent.id in alive_before) and (not agent.alive)

            if just_died:
                r = self.cfg.reward_starvation
                done = True
                self.total_deaths += 1
            elif agent.alive:
                # reward for being alive this step; eating reward was "implicit"
                # via the hunger gain — but we explicitly reward the eat event:
                # handled below via eat tracking
                r = self.cfg.reward_step_alive
                done = False
                agent.steps_survived += 1
            else:
                # already dead from a previous step — skip output collection
                observations.append(np.zeros(self.cfg.obs_dim, dtype=np.float32))
                rewards.append(0.0)
                dones.append(True)
                continue

            obs = self._observe(agent)
            r = float(np.clip(r, self.cfg.reward_clip_min, self.cfg.reward_clip_max))
            observations.append(obs)
            rewards.append(r)
            dones.append(done)

        self.population_history.append(sum(1 for a in self.agents if a.alive))
        return observations, rewards, dones

    def step_with_eat_rewards(
        self, actions: Dict[int, int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        """Step variant that explicitly tracks +reward_eat when an apple is eaten."""
        self.step_count += 1
        observations: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []

        alive_before = {a.id for a in self.agents if a.alive}
        ate: Dict[int, bool] = {}

        for agent in self.agents:
            if not agent.alive:
                continue
            action = Action(actions.get(agent.id, Action.FORWARD))
            agent.apply_action(action, self.W, self.H)

        for agent in self.agents:
            if not agent.alive:
                continue
            ate[agent.id] = False
            if self.apple_grid[agent.y, agent.x]:
                self.apple_grid[agent.y, agent.x] = False
                agent.eat()
                ate[agent.id] = True
                self.total_apples_eaten += 1

        for agent in self.agents:
            if not agent.alive:
                continue
            agent.decay_hunger()
            agent.check_starvation()

        self._respawn_apples()

        for agent in self.agents:
            just_died = (agent.id in alive_before) and (not agent.alive)

            if just_died:
                r = self.cfg.reward_starvation
                done = True
                self.total_deaths += 1
            elif agent.alive:
                if ate.get(agent.id, False):
                    r = self.cfg.reward_eat
                else:
                    r = self.cfg.reward_step_alive
                done = False
                agent.steps_survived += 1
            else:
                observations.append(np.zeros(self.cfg.obs_dim, dtype=np.float32))
                rewards.append(0.0)
                dones.append(True)
                continue

            obs = self._observe(agent)
            r = float(np.clip(r, self.cfg.reward_clip_min, self.cfg.reward_clip_max))
            observations.append(obs)
            rewards.append(r)
            dones.append(done)

        self.population_history.append(sum(1 for a in self.agents if a.alive))
        return observations, rewards, dones

    # ------------------------------------------------------------------
    # apple respawn

    def _respawn_apples(self) -> None:
        """Randomly respawn apples based on respawn probability."""
        current = int(self.apple_grid.sum())
        target = self.cfg.num_apples
        if current >= target:
            return
        agent_positions = {(a.x, a.y) for a in self.agents if a.alive}
        for y in range(self.H):
            for x in range(self.W):
                if (
                    not self.apple_grid[y, x]
                    and (x, y) not in agent_positions
                    and random.random() < self.cfg.apple_respawn_prob
                ):
                    self.apple_grid[y, x] = True
                    current += 1
                    if current >= target:
                        return

    # ------------------------------------------------------------------
    # observation builder

    def _observe(self, agent: Agent) -> np.ndarray:
        """Build the 17-dim observation vector for one agent.

        Layout: [cell_0 one-hot (4), cell_1 (4), cell_2 (4), cell_3 (4), hunger (1)]
        Cell type one-hot: [empty, apple, wall, agent]
        """
        obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)
        look_ahead = agent.cells_ahead(self.cfg.vision_range, self.W, self.H)
        agent_positions = {(a.x, a.y) for a in self.agents if a.alive and a.id != agent.id}

        for i, (cx, cy, is_wall) in enumerate(look_ahead):
            base = i * NUM_CELL_TYPES
            if is_wall:
                obs[base + CELL_WALL] = 1.0
            elif (cx, cy) in agent_positions:
                obs[base + CELL_AGENT] = 1.0
            elif self.apple_grid[cy, cx]:
                obs[base + CELL_APPLE] = 1.0
            else:
                obs[base + CELL_EMPTY] = 1.0

        obs[-1] = agent.hunger / self.cfg.hunger_max
        return obs

    # ------------------------------------------------------------------
    # episode checks

    def all_dead(self) -> bool:
        return not any(a.alive for a in self.agents)

    def alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    def is_done(self) -> bool:
        return self.all_dead() or self.step_count >= self.cfg.max_steps_per_episode

    # ------------------------------------------------------------------
    # metrics helpers

    def civilization_summary(self) -> Dict:
        total_steps = self.population_history
        return {
            "steps_survived": self.step_count,
            "total_deaths": self.total_deaths,
            "total_apples_eaten": self.total_apples_eaten,
            "final_population": sum(1 for a in self.agents if a.alive),
            "avg_steps_survived": (
                sum(a.steps_survived for a in self.agents) / max(len(self.agents), 1)
            ),
            "peak_population": max(total_steps) if total_steps else 0,
        }
