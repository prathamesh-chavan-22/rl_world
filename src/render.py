from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from src.agent import Gender, Orientation

if TYPE_CHECKING:
    from src.config import Config
    from src.world import GridWorld

# ── palette ────────────────────────────────────────────────────────────────
BG_COLOR       = (18,  18,  24)
GRID_COLOR     = (35,  35,  45)
APPLE_COLOR    = (60,  200,  60)
MALE_COLOR     = (70,  130, 220)
FEMALE_COLOR   = (220,  90, 160)
DEAD_COLOR     = (60,   60,  70)
HUNGER_BG      = (50,   50,  60)
HUNGER_FULL    = (80,  200, 120)
HUNGER_LOW     = (220,  80,  60)
VISION_TINT    = (255, 255, 150,  22)   # semi-transparent yellow
TEXT_COLOR     = (200, 200, 200)
ACCENT_COLOR   = (255, 200,  50)

# orientation -> (dx, dy) tip of the direction arrow, relative to cell centre
_ORIENT_ARROW = {
    Orientation.NORTH: ( 0, -1),
    Orientation.EAST:  ( 1,  0),
    Orientation.SOUTH: ( 0,  1),
    Orientation.WEST:  (-1,  0),
}


class Renderer:
    """Pygame renderer for the RL Box World."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.cs = cfg.cell_size          # pixels per cell

        pygame.init()
        pygame.font.init()
        self._font_sm = pygame.font.SysFont("monospace", 11)
        self._font_md = pygame.font.SysFont("monospace", 13, bold=True)

        self._info_panel_w = 220
        self._world_w = cfg.grid_width  * self.cs
        self._world_h = cfg.grid_height * self.cs
        self._win_w   = self._world_w + self._info_panel_w
        self._win_h   = self._world_h

        self.screen = pygame.display.set_mode((self._win_w, self._win_h))
        pygame.display.set_caption("RL Box World")
        self.clock = pygame.time.Clock()

        # surface for the semi-transparent vision cone
        self._vis_surf = pygame.Surface((self.cs, self.cs), pygame.SRCALPHA)
        self._vis_surf.fill(VISION_TINT)

    # ------------------------------------------------------------------
    # public API

    def draw(
        self,
        world: "GridWorld",
        episode: int,
        total_steps: int,
        epsilon: float,
        recent_loss: float | None,
        recent_reward: float | None,
    ) -> bool:
        """Render one frame. Returns False if the window was closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_vision_cones(world)
        self._draw_apples(world)
        self._draw_agents(world)
        self._draw_info_panel(world, episode, total_steps, epsilon, recent_loss, recent_reward)

        pygame.display.flip()
        self.clock.tick(self.cfg.fps)
        return True

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # internal drawing helpers

    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(x * self.cs, y * self.cs, self.cs, self.cs)

    def _draw_grid(self) -> None:
        for x in range(self.cfg.grid_width + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (x * self.cs, 0), (x * self.cs, self._world_h),
            )
        for y in range(self.cfg.grid_height + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (0, y * self.cs), (self._world_w, y * self.cs),
            )

    def _draw_vision_cones(self, world: "GridWorld") -> None:
        for agent in world.alive_agents():
            cells = agent.cells_ahead(self.cfg.vision_range, world.W, world.H)
            for cx, cy, is_wall in cells:
                if not is_wall:
                    self.screen.blit(self._vis_surf, (cx * self.cs, cy * self.cs))

    def _draw_apples(self, world: "GridWorld") -> None:
        r = max(3, self.cs // 4)
        for y in range(world.H):
            for x in range(world.W):
                if world.apple_grid[y, x]:
                    cx = x * self.cs + self.cs // 2
                    cy = y * self.cs + self.cs // 2
                    pygame.draw.circle(self.screen, APPLE_COLOR, (cx, cy), r)

    def _draw_agents(self, world: "GridWorld") -> None:
        pad = 4
        bar_h = 4
        bar_w = self.cs - pad * 2

        for agent in world.agents:
            rect = self._cell_rect(agent.x, agent.y)

            if not agent.alive:
                # faint dead box
                s = pygame.Surface((self.cs, self.cs), pygame.SRCALPHA)
                s.fill((*DEAD_COLOR, 80))
                self.screen.blit(s, rect.topleft)
                continue

            color = MALE_COLOR if agent.gender == Gender.MALE else FEMALE_COLOR
            inner = rect.inflate(-pad, -pad)
            pygame.draw.rect(self.screen, color, inner, border_radius=3)

            # orientation arrow
            dx, dy = _ORIENT_ARROW[agent.orientation]
            cx = rect.centerx
            cy = rect.centery
            arrow_len = self.cs // 3
            tip_x = cx + dx * arrow_len
            tip_y = cy + dy * arrow_len
            pygame.draw.line(self.screen, BG_COLOR, (cx, cy), (tip_x, tip_y), 2)
            # arrowhead dot
            pygame.draw.circle(self.screen, BG_COLOR, (tip_x, tip_y), 3)

            # hunger bar (bottom of cell)
            if self.cfg.render_hunger_bar:
                bar_rect = pygame.Rect(
                    rect.x + pad,
                    rect.bottom - pad - bar_h,
                    bar_w,
                    bar_h,
                )
                pygame.draw.rect(self.screen, HUNGER_BG, bar_rect, border_radius=1)
                fill_w = int(bar_w * (agent.hunger / self.cfg.hunger_max))
                if fill_w > 0:
                    ratio = agent.hunger / self.cfg.hunger_max
                    bar_color = _lerp_color(HUNGER_LOW, HUNGER_FULL, ratio)
                    fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_h)
                    pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=1)

    def _draw_info_panel(
        self,
        world: "GridWorld",
        episode: int,
        total_steps: int,
        epsilon: float,
        recent_loss: float | None,
        recent_reward: float | None,
    ) -> None:
        px = self._world_w + 10
        py = 10
        line_h = 18

        def text(msg: str, color: tuple = TEXT_COLOR, bold: bool = False) -> None:
            nonlocal py
            font = self._font_md if bold else self._font_sm
            surf = font.render(msg, True, color)
            self.screen.blit(surf, (px, py))
            py += line_h

        def gap() -> None:
            nonlocal py
            py += 6

        text("RL BOX WORLD", ACCENT_COLOR, bold=True)
        gap()
        text(f"Episode : {episode}")
        text(f"Ep step : {world.step_count}")
        text(f"Total   : {total_steps}")
        gap()
        text("-- Population --", ACCENT_COLOR, bold=True)
        alive = sum(1 for a in world.agents if a.alive)
        total = len(world.agents)
        text(f"Alive : {alive} / {total}")
        males  = sum(1 for a in world.agents if a.alive and a.gender == Gender.MALE)
        females = sum(1 for a in world.agents if a.alive and a.gender == Gender.FEMALE)
        text(f"M: {males}  F: {females}")
        gap()
        text("-- Resources --", ACCENT_COLOR, bold=True)
        text(f"Apples on grid: {int(world.apple_grid.sum())}")
        text(f"Eaten (ep): {world.total_apples_eaten}")
        gap()
        text("-- Training --", ACCENT_COLOR, bold=True)
        text(f"Epsilon: {epsilon:.3f}")
        loss_str = f"{recent_loss:.4f}" if recent_loss is not None else "—"
        text(f"Loss   : {loss_str}")
        rew_str = f"{recent_reward:.3f}" if recent_reward is not None else "—"
        text(f"Reward : {rew_str}")
        gap()
        text("-- Legend --", ACCENT_COLOR, bold=True)
        text(" Male agents",   MALE_COLOR)
        text(" Female agents", FEMALE_COLOR)
        text(" Apples",        APPLE_COLOR)
        gap()
        text("[ESC] quit", TEXT_COLOR)


# ---------------------------------------------------------------------------
# utility

def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
