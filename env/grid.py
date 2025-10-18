# ===================== env/grid.py =====================
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Position:
    x: int
    y: int

class GridWorld:
    def __init__(self, width: int, height: int, rng: np.random.Generator):
        self.width = width
        self.height = height
        self.rng = rng

    def random_position(self) -> Position:
        return Position(int(self.rng.integers(0, self.width)), int(self.rng.integers(0, self.height)))

    def step_random_move(self, pos: Position) -> Position:
        # 4-neighborhood random step with wrap-around
        dx, dy = self.rng.choice([-1, 0, 1]), self.rng.choice([-1, 0, 1])
        # Disallow standing still to keep motion
        if dx == 0 and dy == 0:
            dx = self.rng.choice([-1, 1])
        nx = (pos.x + dx) % self.width
        ny = (pos.y + dy) % self.height
        return Position(nx, ny)

    @staticmethod
    def in_radius(a: Position, b: Position, radius: int) -> bool:
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 <= radius ** 2
