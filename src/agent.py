from __future__ import annotations
from enum import IntEnum
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config


class Orientation(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# direction vectors for (NORTH, EAST, SOUTH, WEST)
_DELTAS = {
    Orientation.NORTH: (0, -1),
    Orientation.EAST:  (1,  0),
    Orientation.SOUTH: (0,  1),
    Orientation.WEST:  (-1, 0),
}

# turn-left / turn-right lookup
_TURN_LEFT  = [Orientation.WEST, Orientation.NORTH, Orientation.EAST, Orientation.SOUTH]
_TURN_RIGHT = [Orientation.EAST, Orientation.SOUTH, Orientation.WEST, Orientation.NORTH]


class Gender(IntEnum):
    MALE = 0
    FEMALE = 1


class Action(IntEnum):
    FORWARD  = 0
    BACKWARD = 1
    TURN_LEFT  = 2
    TURN_RIGHT = 3


class Agent:
    """A single box agent on the grid."""

    _id_counter: int = 0

    def __init__(
        self,
        x: int,
        y: int,
        orientation: Orientation,
        gender: Gender,
        config: "Config",
    ) -> None:
        Agent._id_counter += 1
        self.id: int = Agent._id_counter

        self.x: int = x
        self.y: int = y
        self.orientation: Orientation = orientation
        self.gender: Gender = gender
        self.config = config

        self.hunger: float = config.hunger_max
        self.alive: bool = True

        # stats for metrics
        self.steps_survived: int = 0
        self.apples_eaten: int = 0

    # ------------------------------------------------------------------
    # movement

    def apply_action(self, action: Action, grid_w: int, grid_h: int) -> None:
        """Apply one action, updating position/orientation in-place."""
        if not self.alive:
            return

        if action == Action.TURN_LEFT:
            self.orientation = _TURN_LEFT[self.orientation]
        elif action == Action.TURN_RIGHT:
            self.orientation = _TURN_RIGHT[self.orientation]
        elif action == Action.FORWARD:
            dx, dy = _DELTAS[self.orientation]
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                self.x, self.y = nx, ny
        elif action == Action.BACKWARD:
            dx, dy = _DELTAS[self.orientation]
            nx, ny = self.x - dx, self.y - dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                self.x, self.y = nx, ny

    # ------------------------------------------------------------------
    # vision

    def cells_ahead(self, n: int, grid_w: int, grid_h: int) -> List[Tuple[int, int, bool]]:
        """Return up to n cell positions directly ahead.

        Each entry is (col, row, is_wall) where is_wall=True when the
        projected position is outside the grid bounds.
        """
        dx, dy = _DELTAS[self.orientation]
        cells = []
        for i in range(1, n + 1):
            cx, cy = self.x + dx * i, self.y + dy * i
            out_of_bounds = not (0 <= cx < grid_w and 0 <= cy < grid_h)
            cells.append((cx, cy, out_of_bounds))
        return cells

    # ------------------------------------------------------------------
    # hunger / lifecycle

    def decay_hunger(self) -> None:
        self.hunger = max(0.0, self.hunger - self.config.hunger_decay)

    def eat(self) -> None:
        self.hunger = min(self.config.hunger_max, self.hunger + self.config.hunger_eat_gain)
        self.apples_eaten += 1

    def is_alive(self) -> bool:
        return self.alive

    def check_starvation(self) -> bool:
        """Return True if the agent just starved (first time hunger == 0)."""
        if self.alive and self.hunger <= 0.0:
            self.alive = False
            return True
        return False

    # ------------------------------------------------------------------
    # reproduction groundwork

    def can_mate(self) -> bool:
        """Placeholder: returns True when conditions for mating are met."""
        return self.alive and self.hunger > self.config.hunger_max * 0.5

    def __repr__(self) -> str:
        g = "M" if self.gender == Gender.MALE else "F"
        o = self.orientation.name[0]
        return (
            f"Agent(id={self.id}, {g}, ({self.x},{self.y}), "
            f"facing={o}, hunger={self.hunger:.2f}, alive={self.alive})"
        )
