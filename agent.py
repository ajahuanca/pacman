from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import QLearningConfig, State


@dataclass
class QLearningAgent:
    config: QLearningConfig
    q_table: Dict[Tuple[State, int], float] = field(default_factory=dict)

    def get_q(self, state: State, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state: State, action: int, value: float) -> None:
        self.q_table[(state, action)] = float(value)

    def get_all_q_for_state(self, state: State) -> List[float]:
        return [self.get_q(state, a) for a in range(4)]

    def best_action(self, state: State) -> int:
        q_values = self.get_all_q_for_state(state)
        max_q = max(q_values)
        candidates = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(candidates)

    def choose_action(self, state: State) -> tuple[int, str]:
        if random.random() < self.config.epsilon:
            return random.randint(0, 3), "Problem Generator (exploración)"
        return self.best_action(state), "Performance Element (explotación)"

    def learn(self, state: State, action: int, reward: float, next_state: State) -> tuple[float, float, float, float]:
        old_q = self.get_q(state, action)
        max_next_q = max(self.get_all_q_for_state(next_state))
        target = reward + self.config.gamma * max_next_q
        new_q = old_q + self.config.alpha * (target - old_q)
        self.set_q(state, action, new_q)
        return old_q, max_next_q, target, new_q

    def reset_q_table(self) -> None:
        self.q_table.clear()
