from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

from config import QLearningConfig, State


@dataclass
class StepResult:
    next_state: State
    reward: float
    done: bool
    event: str


class GridWorldEnv:
    ACTIONS = {
        0: (-1, 0),  # arriba
        1: (1, 0),   # abajo
        2: (0, -1),  # izquierda
        3: (0, 1),   # derecha
    }

    ACTION_NAMES = {
        0: "arriba",
        1: "abajo",
        2: "izquierda",
        3: "derecha",
    }

    def __init__(self, config: QLearningConfig) -> None:
        self.config = config
        self.agent_pos: State = self.random_valid_start()

    def is_inside(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.config.rows and 0 <= c < self.config.cols

    def is_wall(self, state: State) -> bool:
        return state in self.config.wall_positions

    def is_trap(self, state: State) -> bool:
        return state in self.config.trap_positions

    def is_food(self, state: State) -> bool:
        return state == self.config.food_pos

    def is_terminal(self, state: State) -> bool:
        return self.is_food(state) or self.is_trap(state)

    def random_valid_start(self) -> State:
        positions = [
            (r, c)
            for r in range(self.config.rows)
            for c in range(self.config.cols)
            if (r, c) != self.config.food_pos
            and (r, c) not in self.config.trap_positions
            and (r, c) not in self.config.wall_positions
        ]
        return random.choice(positions)

    def reset(self) -> State:
        self.agent_pos = self.random_valid_start()
        return self.agent_pos

    def step(self, action: int) -> StepResult:
        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        proposed = (r + dr, c + dc)

        if not self.is_inside(proposed) or self.is_wall(proposed):
            return StepResult(
                next_state=self.agent_pos,
                reward=self.config.reward_wall_bump,
                done=False,
                event="Choque con pared/límite",
            )

        self.agent_pos = proposed

        if self.is_food(proposed):
            return StepResult(
                next_state=proposed,
                reward=self.config.reward_food,
                done=True,
                event="Objetivo alcanzado",
            )

        if self.is_trap(proposed):
            return StepResult(
                next_state=proposed,
                reward=self.config.reward_trap,
                done=True,
                event="Caída en obstáculo",
            )

        return StepResult(
            next_state=proposed,
            reward=self.config.reward_step,
            done=False,
            event="Paso vacío",
        )
