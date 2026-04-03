from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import QLearningConfig, State


@dataclass
class QLearningAgent:
    """
    Agente de aprendizaje basado en Q-Learning.
    Esta clase encapsula la lógica principal del agente inteligente encargado de:
    - almacenar y administrar la tabla Q,
    - consultar valores Q por estado y acción,
    - seleccionar acciones mediante una política ε-greedy,
    - actualizar el conocimiento aprendido utilizando la ecuación de Q-Learning,
    - reiniciar el aprendizaje cuando sea necesario.
    Atributos:
        config (QLearningConfig):
            Objeto de configuración que contiene los hiperparámetros del algoritmo,
            como la tasa de aprendizaje (alpha), el factor de descuento (gamma)
            y la probabilidad de exploración (epsilon).

        q_table (Dict[Tuple[State, int], float]):
            Tabla Q representada como un diccionario, donde la clave es una tupla
            compuesta por (estado, acción) y el valor es la utilidad estimada
            para dicha combinación.
    Notas:
        - Este agente asume un espacio de acciones discreto de 4 acciones posibles.
        - Cuando un par (estado, acción) no existe en la tabla Q, se considera
          que su valor inicial es 0.0.
    """
    config: QLearningConfig
    q_table: Dict[Tuple[State, int], float] = field(default_factory=dict)

    def get_q(self, state: State, action: int) -> float:
        """
        Obtiene el valor Q asociado a un estado y una acción específica.
        Args:
            state (State):
                Estado actual del entorno.
            action (int):
                Acción cuya utilidad se desea consultar.
        Returns:
            float:
                Valor Q registrado para el par (estado, acción). Si no existe
                en la tabla, retorna 0.0 como valor por defecto.
        Propósito:
            Permitir consultar el conocimiento actual del agente sin necesidad
            de validar previamente si la entrada existe en la tabla Q.
        """
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state: State, action: int, value: float) -> None:
        """
        Registra o actualiza el valor Q de un par (estado, acción).
        Args:
            state (State):
                Estado sobre el cual se actualizará el valor.
            action (int):
                Acción asociada al estado.
            value (float):
                Nuevo valor Q que se desea almacenar.
        Returns:
            None
        Propósito:
            Centralizar la actualización de la tabla Q, asegurando que los valores
            se almacenen de manera consistente como tipo flotante.
        """
        self.q_table[(state, action)] = float(value)

    def get_all_q_for_state(self, state: State) -> List[float]:
        """
        Obtiene los valores Q de todas las acciones posibles para un estado dado.
        Args:
            state (State):
                Estado actual del entorno.
        Returns:
            List[float]:
                Lista de valores Q correspondientes a las acciones 0, 1, 2 y 3.
        Propósito:
            Facilitar la evaluación completa de las acciones disponibles desde
            un estado determinado, lo cual es necesario para seleccionar la mejor
            acción o calcular el valor máximo futuro.
        Nota:
            Este método asume que el agente dispone de exactamente 4 acciones
            discretas posibles.
        """
        return [self.get_q(state, a) for a in range(4)]

    def best_action(self, state: State) -> int:
        """
        Determina la mejor acción disponible para un estado según la tabla Q.
        Args:
            state (State):
                Estado actual del entorno.
        Returns:
            int:
                Acción con el mayor valor Q estimado. Si existen múltiples acciones
                empatadas con el mismo valor máximo, se selecciona una aleatoriamente.
        Propósito:
            Implementar la fase de explotación del agente, es decir, elegir la
            acción que actualmente parece más conveniente según lo aprendido.
        Nota:
            El desempate aleatorio evita sesgos fijos cuando varias acciones
            tienen exactamente la misma utilidad estimada.
        """
        q_values = self.get_all_q_for_state(state)
        max_q = max(q_values)
        candidates = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(candidates)

    def choose_action(self, state: State) -> tuple[int, str]:
        """
        Selecciona una acción utilizando una política ε-greedy.
        La política ε-greedy equilibra dos comportamientos:
        - exploración: probar acciones aleatorias para descubrir nuevas estrategias,
        - explotación: elegir la mejor acción conocida hasta el momento.
        Args:
            state (State):
                Estado actual del entorno.
        Returns:
            tuple[int, str]:
                Una tupla compuesta por:
                - la acción seleccionada,
                - una descripción textual del criterio utilizado:
                  "Problem Generator (exploración)" o
                  "Performance Element (explotación)".
        Propósito:
            Permitir que el agente continúe aprendiendo del entorno mientras
            aprovecha el conocimiento acumulado en la tabla Q.
        """
        if random.random() < self.config.epsilon:
            return random.randint(0, 3), "Problem Generator (exploración)"
        return self.best_action(state), "Performance Element (explotación)"

    def learn(self, state: State, action: int, reward: float, next_state: State) -> tuple[float, float, float, float]:
        """
        Actualiza la tabla Q aplicando la ecuación de aprendizaje de Q-Learning.
        Fórmula aplicada:
            Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
        Donde:
            - Q(s, a): valor actual del estado y acción,
            - alpha: tasa de aprendizaje,
            - reward: recompensa recibida,
            - gamma: factor de descuento,
            - max(Q(s', a')): mejor valor esperado en el siguiente estado.
        Args:
            state (State):
                Estado actual antes de ejecutar la acción.
            action (int):
                Acción ejecutada en el estado actual.
            reward (float):
                Recompensa inmediata recibida después de ejecutar la acción.
            next_state (State):
                Nuevo estado alcanzado después de ejecutar la acción.
        Returns:
            tuple[float, float, float, float]:
                Retorna una tupla con los valores intermedios del proceso:
                - old_q: valor Q anterior,
                - max_next_q: mejor valor Q del siguiente estado,
                - target: objetivo calculado,
                - new_q: nuevo valor Q actualizado.
        Propósito:
            Incorporar la experiencia obtenida por el agente en la tabla Q,
            permitiendo mejorar progresivamente la calidad de las decisiones.
        Importancia:
            Este método constituye el núcleo del aprendizaje del agente, ya que
            transforma recompensas observadas en conocimiento reutilizable para
            episodios futuros.
        """
        old_q = self.get_q(state, action)
        max_next_q = max(self.get_all_q_for_state(next_state))
        target = reward + self.config.gamma * max_next_q
        new_q = old_q + self.config.alpha * (target - old_q)
        self.set_q(state, action, new_q)
        return old_q, max_next_q, target, new_q

    def reset_q_table(self) -> None:
        """
        Reinicia completamente la tabla Q del agente.
        Returns:
            None
        Propósito:
            Eliminar todos los valores aprendidos hasta el momento, dejando
            al agente en un estado limpio de aprendizaje.
        Uso típico:
            - reiniciar una sesión de entrenamiento,
            - comenzar una nueva simulación desde cero,
            - comparar resultados entre distintos experimentos.
        """
        self.q_table.clear()
