import tkinter as tk

from gui import PacmanQLearningGUI


def main() -> None:
    """
    Función principal de arranque de la aplicación.

    Esta función se encarga de inicializar la ventana raíz de Tkinter,
    instanciar la interfaz gráfica principal del sistema y poner en
    ejecución el ciclo de eventos de la aplicación.

    Flujo:
        1. Crea la ventana principal de Tkinter.
        2. Inicializa la interfaz `PacmanQLearningGUI`.
        3. Inicia el loop principal de eventos con `mainloop()`.

    Returns:
        None

    Propósito:
        Servir como punto de entrada formal de la aplicación de escritorio,
        separando la lógica de arranque de la definición de clases y
        facilitando una estructura más limpia y mantenible.
    """
    root = tk.Tk()
    PacmanQLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    """
    Punto de entrada de ejecución directa del módulo.

    Esta condición garantiza que la función `main()` solo se ejecute cuando
    el archivo sea lanzado directamente como programa principal, evitando
    su ejecución automática cuando el módulo sea importado desde otro archivo.

    Propósito:
        Permitir reutilización del módulo sin efectos colaterales y seguir
        una práctica estándar de organización en aplicaciones Python.
    """
    main()
