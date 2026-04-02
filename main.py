import tkinter as tk

from gui import PacmanQLearningGUI


def main() -> None:
    root = tk.Tk()
    PacmanQLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
