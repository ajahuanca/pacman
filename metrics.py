from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt


@dataclass
class TrainingMetrics:
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    wins: int = 0
    trap_hits: int = 0
    max_step_terminations: int = 0

    def record(self, reward: float, steps: int, outcome: str) -> None:
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)

        if outcome == "win":
            self.wins += 1
        elif outcome == "trap":
            self.trap_hits += 1
        elif outcome == "max_steps":
            self.max_step_terminations += 1

    def reset(self) -> None:
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.wins = 0
        self.trap_hits = 0
        self.max_step_terminations = 0

    def summary(self) -> str:
        total = len(self.episode_rewards)
        avg_reward = sum(self.episode_rewards) / total if total else 0.0
        avg_steps = sum(self.episode_steps) / total if total else 0.0
        win_rate = (self.wins / total * 100.0) if total else 0.0

        return (
            f"Episodios registrados: {total}\n"
            f"Victorias: {self.wins}\n"
            f"Caídas en obstáculo: {self.trap_hits}\n"
            f"Terminaciones por límite de pasos: {self.max_step_terminations}\n"
            f"Recompensa promedio: {avg_reward:.3f}\n"
            f"Pasos promedio: {avg_steps:.2f}\n"
            f"Tasa de éxito: {win_rate:.2f}%"
        )

    def save_convergence_chart(self, filepath: str) -> None:
        if not self.episode_rewards:
            raise ValueError("No hay métricas para graficar.")

        window = 20
        rolling = []
        for i in range(len(self.episode_rewards)):
            start = max(0, i - window + 1)
            segment = self.episode_rewards[start:i + 1]
            rolling.append(sum(segment) / len(segment))

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Recompensa por episodio")
        plt.plot(rolling, label=f"Promedio móvil ({window})")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Convergencia del entrenamiento Q-Learning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
