from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

from config import QLearningConfig, State


@dataclass
class StepResult:
    """
    Estructura de datos que representa el resultado de ejecutar una acción
    dentro del entorno.

    Esta clase se utiliza para encapsular de forma clara y consistente la
    respuesta del entorno después de que el agente realiza un movimiento.

    Atributos:
        next_state (State):
            Estado resultante luego de ejecutar la acción.

        reward (float):
            Recompensa o penalización obtenida por la transición realizada.

        done (bool):
            Indicador que señala si el episodio ha finalizado. Normalmente
            será verdadero cuando el agente alcance el objetivo o caiga
            en una trampa.

        event (str):
            Descripción textual del evento ocurrido durante la transición.
            Puede utilizarse para trazabilidad, depuración o visualización
            en la interfaz gráfica.
    """
    next_state: State
    reward: float
    done: bool
    event: str


class GridWorldEnv:
    """
    Entorno tipo grilla para el entrenamiento del agente con Q-Learning.

    Esta clase modela el escenario donde se mueve el agente, incluyendo:
    - dimensiones del tablero,
    - movimientos válidos,
    - muros,
    - trampas,
    - objetivo final,
    - reinicio de episodios,
    - cálculo de recompensas por acción.

    Propósito:
        Proveer un entorno discreto y controlado donde el agente pueda
        interactuar, recibir retroalimentación y aprender una política
        óptima para alcanzar el objetivo evitando obstáculos y trampas.

    Atributos de clase:
        ACTIONS:
            Diccionario que mapea cada acción a un desplazamiento en la grilla.
            Cada valor representa el cambio en fila y columna.

        ACTION_NAMES:
            Diccionario que asocia cada acción con un nombre descriptivo,
            útil para depuración, monitoreo o interfaz visual.
    """

    ACTIONS = {
        0: (-1, 0),  # arriba
        1: (1, 0),   # abajo
        2: (0, -1),  # izquierda
        3: (0, 1),   # derecha
    }
    """
    Mapeo de acciones a desplazamientos dentro del tablero.
    Convenciones:
        0 -> mover arriba
        1 -> mover abajo
        2 -> mover izquierda
        3 -> mover derecha
    """

    ACTION_NAMES = {
        0: "arriba",
        1: "abajo",
        2: "izquierda",
        3: "derecha",
    }
    """
    Nombres descriptivos de las acciones disponibles.
    Esta estructura facilita la interpretación humana de los movimientos
    realizados por el agente.
    """

    def __init__(self, config: QLearningConfig) -> None:
        """
        Inicializa el entorno de la grilla con la configuración recibida.
        Args:
            config (QLearningConfig):
                Objeto de configuración que define el tamaño del tablero,
                recompensas, posición de la comida, trampas y muros.
        Returns:
            None
        Propósito:
            Preparar el entorno para su uso, almacenando la configuración
            y asignando una posición inicial válida y aleatoria al agente.
        """
        self.config = config
        self.agent_pos: State = self.random_valid_start()

    def is_inside(self, state: State) -> bool:
        """
        Verifica si una posición se encuentra dentro de los límites válidos del tablero.
        Args:
            state (State):
                Estado o posición a validar, representado como (fila, columna).
        Returns:
            bool:
                True si la posición está dentro de la grilla; False en caso contrario.
        Propósito:
            Evitar que el agente se desplace fuera de los límites permitidos del entorno.
        """
        r, c = state
        return 0 <= r < self.config.rows and 0 <= c < self.config.cols

    def is_wall(self, state: State) -> bool:
        """
        Verifica si una posición corresponde a un muro o casilla bloqueada.
        Args:
            state (State):
                Estado o posición a evaluar.
        Returns:
            bool:
                True si la posición está definida como muro; False si no lo está.
        Propósito:
            Determinar si el agente puede o no desplazarse a una casilla
            específica del tablero.
        """
        return state in self.config.wall_positions

    def is_trap(self, state: State) -> bool:
        """
        Verifica si una posición corresponde a una trampa.
        Args:
            state (State):
                Estado o posición a evaluar.
        Returns:
            bool:
                True si la posición pertenece al conjunto de trampas;
                False en caso contrario.
        Propósito:
            Identificar estados que finalizan negativamente el episodio
            y generan una penalización para el agente.
        """
        return state in self.config.trap_positions

    def is_food(self, state: State) -> bool:
        """
        Verifica si una posición coincide con la ubicación del objetivo.
        Args:
            state (State):
                Estado o posición a evaluar.
        Returns:
            bool:
                True si la posición corresponde a la comida; False en caso
                contrario.
        Propósito:
            Determinar si el agente ha alcanzado el objetivo principal
            del entorno.
        """
        return state == self.config.food_pos

    def is_terminal(self, state: State) -> bool:
        """
        Verifica si un estado es terminal.
        Un estado terminal es aquel que finaliza el episodio, ya sea porque
        el agente alcanzó la comida o porque cayó en una trampa.
        Args:
            state (State):
                Estado o posición a evaluar.
        Returns:
            bool:
                True si el estado es terminal; False si el episodio puede
                continuar.
        Propósito:
            Facilitar la identificación de estados finales dentro del flujo
            de entrenamiento o simulación.
        """
        return self.is_food(state) or self.is_trap(state)

    def random_valid_start(self) -> State:
        """
        Genera una posición inicial aleatoria válida para el agente.
        La posición inicial debe cumplir con las siguientes restricciones:
        - no ser la posición de la comida,
        - no ser una trampa,
        - no ser un muro.
        Args:
            None
        Returns:
            State:
                Una posición válida seleccionada aleatoriamente dentro
                del tablero.
        Propósito:
            Garantizar que cada episodio comience en una casilla transitable
            y no terminal, permitiendo diversidad en el entrenamiento.
        """
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
        """
        Reinicia el entorno para comenzar un nuevo episodio.
        Este método asigna una nueva posición inicial válida y aleatoria al agente.
        Args:
            None
        Returns:
            State:
                Nueva posición inicial del agente.
        Propósito:
            Restablecer el entorno entre episodios de entrenamiento o
            simulación, manteniendo variabilidad en la posición inicial.
        """
        self.agent_pos = self.random_valid_start()
        return self.agent_pos

    def step(self, action: int) -> StepResult:
        """
        Ejecuta una acción en el entorno y retorna el resultado de la transición.
        Flujo general:
            1. Calcula la posición propuesta a partir de la acción.
            2. Verifica si el movimiento es inválido por límite o muro.
            3. Si es válido, actualiza la posición del agente.
            4. Evalúa si la nueva posición corresponde a comida, trampa
               o casilla vacía.
            5. Devuelve un objeto StepResult con el nuevo estado, recompensa,
               indicador de fin y descripción del evento.
        Args:
            action (int):
                Acción discreta que el agente desea ejecutar.
        Returns:
            StepResult:
                Objeto con el detalle completo de la transición realizada.
        Propósito:
            Modelar la dinámica del entorno, definiendo cómo responde el
            sistema ante cada acción del agente y qué retroalimentación
            se entrega para el proceso de aprendizaje.
        Nota:
            Este método asume que la acción recibida es válida y existe
            dentro del diccionario ACTIONS.
        """
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
