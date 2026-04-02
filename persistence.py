from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from config import State


def save_q_table(q_table: Dict[Tuple[State, int], float], filepath: str) -> None:
    serialized = {
        f"{state[0]},{state[1]}|{action}": value
        for (state, action), value in q_table.items()
    }
    Path(filepath).write_text(json.dumps(serialized, indent=4), encoding="utf-8")


def load_q_table(filepath: str) -> Dict[Tuple[State, int], float]:
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
