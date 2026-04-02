# Pac-Man Q-Learning

Ejercicio de Pac-Man con Q-Learning.

## Características

- Pac-Man inicia en posición aleatoria válida.
- Objetivo fijo.
- Múltiples obstáculos fijos.
- Paredes reales que bloquean movimiento.
- Persistencia de tabla Q en JSON.
- Métricas de entrenamiento.
- Exportación de gráfico de convergencia.
- GUI en Tkinter con enfoque didáctico y visual.

## Arquitectura

### `config.py`
Parámetros centralizados del entorno y del algoritmo.

### `environment.py`
Lógica del tablero:
- límites,
- paredes,
- obstáculos,
- objetivo,
- recompensas,
- transición de estados.

### `agent.py`
Implementación del agente:
- tabla Q,
- selección ε-greedy,
- actualización Q-learning.

### `metrics.py`
Registro de desempeño por episodio y exportación de convergencia.

### `persistence.py`
Guardado y carga de la tabla Q en JSON.

### `gui.py`
Interfaz gráfica, visualización, log y controles operativos.

### `main.py`
Punto de entrada.

## Requisitos

- Python 3.12 o superior
- Tkinter
- matplotlib

## Instalación

```bash
pip install matplotlib
```
o crear un entorno virtual especifico para este proyecto

```bash
py -3.12 -m venv envPacman
-- activando entorno virtual
envPacman/Scripts/activate
pip install -r requirements.txt
```
## Ejecución

Desde la carpeta del proyecto:

```bash
python main.py
```

## Controles

- **Paso**: ejecuta una iteración.
- **Episodio**: ejecuta hasta terminar.
- **Entrenar x100/x500**: entrenamiento masivo.
- **Auto lento**: demostración paso a paso.
- **Política aprendida**: explota solo la mejor acción conocida.
- **Guardar Q JSON**: serializa la tabla Q.
- **Cargar Q JSON**: restaura la tabla Q.
- **Gráfico convergencia**: exporta métricas a PNG.
- **Reset posición**: reposiciona Pac-Man sin borrar conocimiento.
- **Reset Q**: reinicia aprendizaje y métricas.

## Recompensas

- Objetivo: `+1.0`
- Obstáculo: `-1.0`
- Paso vacío: `-0.01`
- Choque con pared/límite: `-0.05`

