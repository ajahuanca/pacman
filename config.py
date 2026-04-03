from dataclasses import dataclass, field
from typing import List, Tuple

State = Tuple[int, int]


@dataclass
class QLearningConfig:
    """
    Clase de configuración para el entorno y el algoritmo de Q-Learning.
    Esta clase centraliza los parámetros principales del sistema, incluyendo:
    - dimensiones del tablero,
    - hiperparámetros de aprendizaje,
    - esquema de recompensas,
    - límites de episodios,
    - posiciones fijas de los elementos del entorno.
    Propósito:
        Proveer una única fuente de configuración para que el agente y el
        entorno trabajen con valores consistentes, facilitando el mantenimiento,
        la reutilización y la modificación de parámetros durante pruebas
        o experimentos.
    Atributos:
        rows (int):
            Número de filas de la grilla o tablero.

        cols (int):
            Número de columnas de la grilla o tablero.

        alpha (float):
            Tasa de aprendizaje del algoritmo Q-Learning. Controla cuánto
            influye la nueva experiencia sobre el valor previamente aprendido.

        gamma (float):
            Factor de descuento. Determina la importancia de las recompensas
            futuras respecto a las recompensas inmediatas.

        epsilon (float):
            Probabilidad de exploración utilizada en la política ε-greedy.
            Un valor mayor incrementa la frecuencia con la que el agente
            toma acciones aleatorias.

        reward_food (float):
            Recompensa positiva otorgada cuando el agente alcanza la comida
            u objetivo.

        reward_trap (float):
            Penalización aplicada cuando el agente cae en una trampa
            u obstáculo dañino.

        reward_step (float):
            Penalización pequeña aplicada en cada paso normal para incentivar
            rutas más cortas y eficientes.

        reward_wall_bump (float):
            Penalización aplicada cuando el agente intenta moverse hacia una
            pared o posición inválida y no puede avanzar.

        max_steps_per_episode (int):
            Número máximo de pasos permitidos por episodio para evitar ciclos
            infinitos o recorridos excesivamente largos.

        food_pos (State):
            Posición fija de la comida dentro del tablero.

        trap_positions (List[State]):
            Lista de posiciones donde se ubican trampas o casillas de castigo.

        wall_positions (List[State]):
            Lista de posiciones ocupadas por muros o bloqueos que el agente
            no puede atravesar.
    """

    rows: int = 7
    """
    Número total de filas del tablero.
    Define la dimensión vertical del entorno donde se desplaza el agente.
    """

    cols: int = 7
    """
    Número total de columnas del tablero.
    Define la dimensión horizontal del entorno donde se desplaza el agente.
    """

    alpha: float = 0.10
    """
    Tasa de aprendizaje del algoritmo Q-Learning.
    Indica qué proporción del nuevo conocimiento se incorpora al valor Q actual.
    Valores bajos producen un aprendizaje más gradual y estable.
    """

    gamma: float = 0.90
    """
    Factor de descuento de recompensas futuras.
    Un valor cercano a 1 da mayor importancia a beneficios futuros, favoreciendo
    decisiones estratégicas de largo plazo.
    """

    epsilon: float = 0.20
    """
    Probabilidad de exploración de la política ε-greedy.
    Con esta probabilidad el agente elegirá una acción aleatoria en lugar
    de la mejor acción conocida, promoviendo exploración del entorno.
    """

    reward_food: float = 1.0
    """
    Recompensa positiva por alcanzar el objetivo.
    Este valor incentiva al agente a encontrar caminos que lo lleven hasta la comida.
    """

    reward_trap: float = -1.0
    """
    Penalización por caer en una trampa.
    Este castigo desincentiva rutas peligrosas o estados no deseados.
    """

    reward_step: float = -0.01
    """
    Penalización aplicada por cada movimiento normal.
    Su objetivo es evitar trayectorias innecesariamente largas y estimular
    la búsqueda de caminos más cortos hacia el objetivo.
    """

    reward_wall_bump: float = -0.05
    """
    Penalización aplicada cuando el agente intenta moverse contra un muro
    o fuera de los límites válidos del tablero.
    Este castigo ayuda a que el agente aprenda a evitar movimientos inválidos.
    """

    max_steps_per_episode: int = 80
    """
    Límite máximo de pasos por episodio.
    Se utiliza para cortar episodios que no llegan al objetivo en un tiempo
    razonable, evitando ciclos o exploraciones improductivas.
    """

    food_pos: State = (0, 6)
    """
    Posición fija del objetivo principal del entorno.
    Representa la ubicación de la comida que el agente debe alcanzar.
    """

    trap_positions: List[State] = field(default_factory=lambda: [(3, 3), (5, 2)])
    """
    Lista de posiciones de trampas dentro del tablero.
    Cada coordenada representa una casilla que genera una penalización al ser
    alcanzada por el agente.
    """

    wall_positions: List[State] = field(default_factory=lambda: [(1, 2), (2, 2), (3, 2), (4, 4), (4, 5)])
    """
    Lista de posiciones ocupadas por muros o bloqueos.
    Estas casillas no son transitables y restringen el movimiento del agente
    dentro del entorno.
    """
