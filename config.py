from dataclasses import dataclass, field
from typing import List, Tuple

State = Tuple[int, int]


@dataclass
class QLearningConfig:
    rows: int = 7
    cols: int = 7

    alpha: float = 0.10
    gamma: float = 0.90
    epsilon: float = 0.20

    reward_food: float = 1.0
    reward_trap: float = -1.0
    reward_step: float = -0.01
    reward_wall_bump: float = -0.05

    max_steps_per_episode: int = 80

    food_pos: State = (0, 6)
    trap_positions: List[State] = field(default_factory=lambda: [(3, 3), (5, 2)])
    wall_positions: List[State] = field(default_factory=lambda: [(1, 2), (2, 2), (3, 2), (4, 4), (4, 5)])
