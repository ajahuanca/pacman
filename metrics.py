from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt


@dataclass
class TrainingMetrics:
    """
    Clase encargada de almacenar, resumir y exportar métricas del proceso
    de entrenamiento del agente Q-Learning.

    Esta clase centraliza la información estadística generada durante los
    episodios de entrenamiento, permitiendo:
    - registrar recompensas y pasos por episodio,
    - contabilizar resultados relevantes del agente,
    - generar resúmenes textuales del desempeño,
    - exportar gráficos de convergencia del aprendizaje.

    Atributos:
        episode_rewards (List[float]):
            Lista de recompensas totales obtenidas en cada episodio.

        episode_steps (List[int]):
            Lista con la cantidad de pasos ejecutados por episodio.

        wins (int):
            Contador de episodios finalizados exitosamente al alcanzar
            el objetivo.

        trap_hits (int):
            Contador de episodios finalizados por caída en obstáculo
            o trampa.

        max_step_terminations (int):
            Contador de episodios finalizados por alcanzar el límite
            máximo de pasos permitidos.
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    wins: int = 0
    trap_hits: int = 0
    max_step_terminations: int = 0

    def record(self, reward: float, steps: int, outcome: str) -> None:
        """
        Registra las métricas de un episodio finalizado.
        Args:
            reward (float):
                Recompensa total acumulada durante el episodio.
            steps (int):
                Cantidad total de pasos ejecutados en el episodio.
            outcome (str):
                Resultado final del episodio. Los valores esperados son:
                - "win": el agente alcanzó el objetivo,
                - "trap": el agente cayó en un obstáculo,
                - "max_steps": el episodio terminó por límite de pasos.
        Returns:
            None
        Propósito:
            Almacenar el resultado del episodio y actualizar los contadores
            acumulados según el desenlace alcanzado.
        Nota:
            Este método asume que el parámetro `outcome` llega con uno de
            los valores válidos definidos por la aplicación.
        """
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)

        if outcome == "win":
            self.wins += 1
        elif outcome == "trap":
            self.trap_hits += 1
        elif outcome == "max_steps":
            self.max_step_terminations += 1

    def reset(self) -> None:
        """
        Reinicia todas las métricas acumuladas del entrenamiento.
        Returns:
            None
        Propósito:
            Limpiar completamente el historial estadístico del agente para
            comenzar una nueva sesión de entrenamiento o experimento desde cero.
        Acciones realizadas:
            - vacía las listas de recompensas y pasos,
            - reinicia los contadores de victorias, caídas y terminaciones
              por límite de pasos.
        """
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.wins = 0
        self.trap_hits = 0
        self.max_step_terminations = 0

    def summary(self) -> str:
        """
        Genera un resumen textual consolidado de las métricas actuales.
        Returns:
            str:
                Cadena formateada con indicadores principales del entrenamiento,
                incluyendo:
                - número de episodios registrados,
                - victorias,
                - caídas en obstáculo,
                - terminaciones por límite de pasos,
                - recompensa promedio,
                - pasos promedio,
                - tasa de éxito.
        Propósito:
            Proveer una vista resumida y legible del desempeño global del agente,
            útil para mostrar en la interfaz gráfica o en salidas de monitoreo.
        Nota:
            Cuando no existen episodios registrados, los promedios y la tasa
            de éxito se reportan como 0.0 para evitar divisiones por cero.
        """
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
        """
        Genera y guarda un gráfico de convergencia del entrenamiento.
        El gráfico incluye:
        - la recompensa obtenida en cada episodio,
        - un promedio móvil para suavizar la curva y facilitar el análisis
          del comportamiento de convergencia.
        Args:
            filepath (str):
                Ruta completa donde se guardará la imagen del gráfico.
        Returns:
            None
        Raises:
            ValueError:
                Se lanza cuando no existen métricas registradas para graficar.
        Propósito:
            Exportar una representación visual del progreso de aprendizaje
            del agente, útil para evaluación, documentación y presentación
            de resultados.
        Flujo:
            1. Verifica que existan recompensas registradas.
            2. Calcula un promedio móvil con ventana de 20 episodios.
            3. Genera la figura con matplotlib.
            4. Guarda la imagen en disco.
            5. Cierra la figura para liberar recursos.
        """
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
