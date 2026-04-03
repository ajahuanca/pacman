from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from config import State


def save_q_table(q_table: Dict[Tuple[State, int], float], filepath: str) -> None:
    """
    Guarda la tabla Q en un archivo JSON.

    Debido a que JSON no permite usar tuplas como claves de diccionario,
    este método transforma cada clave `(estado, acción)` a una representación
    serializable en formato de texto.
    Formato de serialización de clave:
        "fila,columna|accion"

    Ejemplo:
        ((2, 3), 1) -> "2,3|1"

    Args:
        q_table (Dict[Tuple[State, int], float]):
            Tabla Q del agente, donde la clave es una tupla compuesta por
            `(estado, acción)` y el valor es la utilidad aprendida.
        filepath (str):
            Ruta del archivo donde se almacenará el contenido en formato JSON.
    Returns:
        None
    Propósito:
        Persistir el conocimiento aprendido por el agente para permitir
        su reutilización posterior sin necesidad de reentrenar desde cero.
    Flujo:
        1. Convierte la tabla Q a un diccionario serializable.
        2. Genera la representación JSON con indentación legible.
        3. Escribe el contenido en disco usando codificación UTF-8.
    Nota:
        Este método sobrescribe el archivo destino si ya existe.
    """
    serialized = {
        f"{state[0]},{state[1]}|{action}": value
        for (state, action), value in q_table.items()
    }
    Path(filepath).write_text(json.dumps(serialized, indent=4), encoding="utf-8")


def load_q_table(filepath: str) -> Dict[Tuple[State, int], float]:
    """
    Carga una tabla Q desde un archivo JSON y reconstruye su estructura original.

    Este método realiza el proceso inverso a `save_q_table`, interpretando
    las claves serializadas en texto para volver a formar claves del tipo
    `(estado, acción)`.
    Formato esperado de clave en el JSON:
        "fila,columna|accion"
    Ejemplo:
        "2,3|1" -> ((2, 3), 1)
    Args:
        filepath (str):
            Ruta del archivo JSON desde el cual se cargará la tabla Q.
    Returns:
        Dict[Tuple[State, int], float]:
            Diccionario reconstruido que representa la tabla Q original.
    Raises:
        FileNotFoundError:
            Se lanza cuando la ruta indicada no existe.
    Propósito:
        Restaurar el conocimiento previamente persistido del agente,
        permitiendo continuar con entrenamiento, pruebas o demostraciones
        desde un estado aprendido.
    Flujo:
        1. Verifica que el archivo exista.
        2. Lee el contenido JSON del archivo.
        3. Reconstruye cada clave serializada a la forma `(estado, acción)`.
        4. Convierte todos los valores a tipo `float`.
        5. Retorna la tabla Q reconstruida.
    Nota:
        Este método asume que el archivo tiene una estructura válida y
        compatible con el formato generado por `save_q_table`.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {filepath}")

    data = json.loads(path.read_text(encoding="utf-8"))
    q_table: Dict[Tuple[State, int], float] = {}

    for key, value in data.items():
        state_part, action_part = key.split("|")
        row, col = state_part.split(",")
        state = (int(row), int(col))
        action = int(action_part)
        q_table[(state, action)] = float(value)

    return q_table
