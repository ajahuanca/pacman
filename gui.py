from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent import QLearningAgent
from config import QLearningConfig, State
from environment import GridWorldEnv, StepResult
from metrics import TrainingMetrics
from persistence import load_q_table, save_q_table


@dataclass
class StepInfo:
    prev_state: State
    action: int
    action_name: str
    next_state: State
    reward: float
    done: bool
    old_q: float
    max_next_q: float
    target: float
    new_q: float
    source: str
    event: str


class PacmanQLearningGUI:
    CELL_SIZE = 72
    GRID_PADDING = 20
    BG = "#f4f7fb"
    PANEL_BG = "#ffffff"
    DARK = "#1f2937"
    MUTED = "#6b7280"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pac-Man Q-Learning")
        self.root.geometry("1460x920")
        self.root.configure(bg=self.BG)
        self.root.minsize(1320, 840)

        self.config = QLearningConfig()
        self.env = GridWorldEnv(self.config)
        self.agent = QLearningAgent(self.config)
        self.metrics = TrainingMetrics()

        self.episode = 1
        self.step_count = 0
        self.total_reward = 0.0
        self.last_info: Optional[StepInfo] = None

        self._build_styles()
        self._build_layout()
        self._draw_grid()
        self._refresh_panels(initial=True)

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Card.TLabelframe", background=self.PANEL_BG, borderwidth=1)
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"), foreground=self.DARK)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background=self.BG, foreground=self.DARK)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), background=self.BG, foreground=self.MUTED)
        style.configure("Info.TLabel", font=("Segoe UI", 10), background=self.PANEL_BG, foreground=self.DARK)
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Secondary.TButton", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=4)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=14)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        right = ttk.Frame(self.root, padding=14)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        ttk.Label(left, text="Pac-Man con Q-Learning", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            left,
            text="Versión modular con paredes, múltiples obstáculos, persistencia JSON, métricas y convergencia.\nEdwin Ajahuanca Callisaya",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        board_card = ttk.LabelFrame(left, text="Tablero", padding=10, style="Card.TLabelframe")
        board_card.grid(row=2, column=0, sticky="nsew")
        board_card.columnconfigure(0, weight=1)
        board_card.rowconfigure(0, weight=1)

        canvas_w = self.config.cols * self.CELL_SIZE + self.GRID_PADDING * 2
        canvas_h = self.config.rows * self.CELL_SIZE + self.GRID_PADDING * 2
        self.canvas = tk.Canvas(board_card, width=canvas_w, height=canvas_h, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(left, text="Acciones", padding=10, style="Card.TLabelframe")
        controls.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for col in range(4):
            controls.columnconfigure(col, weight=1)

        ttk.Button(controls, text="Paso", style="Primary.TButton", command=self.step_once).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Episodio", style="Primary.TButton", command=self.run_one_episode).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Entrenar x100", style="Primary.TButton", command=lambda: self.train_many(100)).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Entrenar x500", style="Primary.TButton", command=lambda: self.train_many(500)).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ttk.Button(controls, text="Auto lento", style="Secondary.TButton", command=self.auto_slow_episode).grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Política aprendida", style="Secondary.TButton", command=self.show_greedy_demo).grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Guardar Q JSON", style="Secondary.TButton", command=self.save_q_json).grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Cargar Q JSON", style="Secondary.TButton", command=self.load_q_json).grid(row=1, column=3, padx=4, pady=4, sticky="ew")

        ttk.Button(controls, text="Gráfico convergencia", style="Secondary.TButton", command=self.export_convergence_chart).grid(row=2, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Reset posición", style="Secondary.TButton", command=self.reset_position_only).grid(row=2, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Reset Q", style="Secondary.TButton", command=self.reset_all).grid(row=2, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(controls, text="Limpiar log", style="Secondary.TButton", command=self.clear_log).grid(row=2, column=3, padx=4, pady=4, sticky="ew")

        self.summary_frame = ttk.LabelFrame(right, text="Resumen operativo", padding=10, style="Card.TLabelframe")
        self.summary_frame.grid(row=0, column=0, sticky="ew")
        self.summary_content = ttk.Label(self.summary_frame, text="", justify="left", style="Info.TLabel")
        self.summary_content.grid(row=0, column=0, sticky="w")

        self.conditions_frame = ttk.LabelFrame(right, text="Condiciones de Q-Learning", padding=10, style="Card.TLabelframe")
        self.conditions_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.conditions_content = ttk.Label(self.conditions_frame, text="", justify="left", style="Info.TLabel")
        self.conditions_content.grid(row=0, column=0, sticky="w")

        self.formula_frame = ttk.LabelFrame(right, text="Aplicación de la fórmula", padding=10, style="Card.TLabelframe")
        self.formula_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.formula_content = ttk.Label(self.formula_frame, text="", justify="left", style="Info.TLabel")
        self.formula_content.grid(row=0, column=0, sticky="w")

        log_card = ttk.LabelFrame(right, text="Log y métricas", padding=10, style="Card.TLabelframe")
        log_card.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)

        self.metrics_content = ttk.Label(log_card, text="", justify="left", style="Info.TLabel")
        self.metrics_content.grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.log_text = ScrolledText(log_card, wrap="word", font=("Consolas", 10), bg="#0b1220", fg="#d1d5db", insertbackground="white")
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        q_card = ttk.LabelFrame(right, text="Q-values del estado actual", padding=10, style="Card.TLabelframe")
        q_card.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.q_content = ttk.Label(q_card, text="", justify="left", style="Info.TLabel")
        self.q_content.grid(row=0, column=0, sticky="w")

    def _cell_to_xy(self, row: int, col: int) -> tuple[int, int, int, int]:
        x1 = self.GRID_PADDING + col * self.CELL_SIZE
        y1 = self.GRID_PADDING + row * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE
        return x1, y1, x2, y2

    def _draw_grid(self) -> None:
        self.canvas.delete("all")

        for r in range(self.config.rows):
            for c in range(self.config.cols):
                x1, y1, x2, y2 = self._cell_to_xy(r, c)
                fill = "#f9fafb"
                if (r, c) in self.config.wall_positions:
                    fill = "#374151"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#cbd5e1")
                self.canvas.create_text(x1 + 8, y1 + 8, anchor="nw", text=f"{r},{c}", fill="#94a3b8", font=("Segoe UI", 8))

        fr, fc = self.config.food_pos
        x1, y1, x2, y2 = self._cell_to_xy(fr, fc)
        self.canvas.create_oval(x1 + 12, y1 + 12, x2 - 12, y2 - 12, fill="#10b981", outline="#047857", width=2)
        self.canvas.create_text((x1 + x2) // 2, y2 - 10, text="Objetivo", fill="#065f46", font=("Segoe UI", 8, "bold"))

        for tr, tc in self.config.trap_positions:
            x1, y1, x2, y2 = self._cell_to_xy(tr, tc)
            self.canvas.create_rectangle(x1 + 12, y1 + 12, x2 - 12, y2 - 12, fill="#ef4444", outline="#991b1b", width=2)
            self.canvas.create_text((x1 + x2) // 2, y2 - 10, text="Obstáculo", fill="#7f1d1d", font=("Segoe UI", 8, "bold"))

        ar, ac = self.env.agent_pos
        x1, y1, x2, y2 = self._cell_to_xy(ar, ac)
        self.canvas.create_oval(x1 + 16, y1 + 16, x2 - 16, y2 - 16, fill="#2563eb", outline="#1e3a8a", width=2)
        self.canvas.create_text((x1 + x2) // 2, y2 - 10, text="Pac-Man", fill="#1e40af", font=("Segoe UI", 8, "bold"))

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _refresh_panels(self, initial: bool = False) -> None:
        state = self.env.agent_pos
        q_values = self.agent.get_all_q_for_state(state)
        best_action = self.agent.best_action(state)

        self.summary_content.config(
            text=(
                f"Episodio actual: {self.episode}\n"
                f"Paso actual: {self.step_count}\n"
                f"Recompensa acumulada: {self.total_reward:.3f}\n"
                f"Posición del agente: {state}\n"
                f"Objetivo fijo: {self.config.food_pos}\n"
                f"Obstáculos fijos: {self.config.trap_positions}\n"
                f"Paredes: {self.config.wall_positions}"
            )
        )

        self.conditions_content.config(
            text=(
                "Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]\n\n"
                f"α = {self.config.alpha}  |  γ = {self.config.gamma}  |  ε = {self.config.epsilon}\n"
                f"Recompensa objetivo: {self.config.reward_food}\n"
                f"Castigo obstáculo: {self.config.reward_trap}\n"
                f"Paso vacío: {self.config.reward_step}\n"
                f"Choque con pared/límite: {self.config.reward_wall_bump}\n"
                f"Grilla: {self.config.rows}x{self.config.cols}\n"
                f"Máximo de pasos por episodio: {self.config.max_steps_per_episode}"
            )
        )

        self.metrics_content.config(text=self.metrics.summary())

        self.q_content.config(
            text=(
                f"Estado actual: {state}\n"
                f"Q(arriba)    = {q_values[0]:.3f}\n"
                f"Q(abajo)     = {q_values[1]:.3f}\n"
                f"Q(izquierda) = {q_values[2]:.3f}\n"
                f"Q(derecha)   = {q_values[3]:.3f}\n"
                f"Mejor acción conocida: {GridWorldEnv.ACTION_NAMES[best_action]}"
            )
        )

        if initial or self.last_info is None:
            self.formula_content.config(
                text=(
                    "Estado inicial:\n"
                    "- Pac-Man inicia en una posición aleatoria válida.\n"
                    "- El objetivo está fijo.\n"
                    "- Los obstáculos y paredes están fijos.\n"
                    "- La tabla Q comienza en cero.\n"
                    "- Aún no existe actualización hasta ejecutar una acción."
                )
            )
            return

        info = self.last_info
        self.formula_content.config(
            text=(
                f"s = {info.prev_state}\n"
                f"a = {info.action} ({info.action_name})\n"
                f"s' = {info.next_state}\n"
                f"Evento = {info.event}\n"
                f"r = {info.reward:.3f}\n"
                f"maxQ(s',a') = {info.max_next_q:.3f}\n"
                f"Q anterior = {info.old_q:.3f}\n\n"
                f"target = r + γ·maxQ(s',a') = {info.reward:.3f} + {self.config.gamma}·{info.max_next_q:.3f} = {info.target:.3f}\n"
                f"Q nuevo = {info.old_q:.3f} + {self.config.alpha}·({info.target:.3f} - {info.old_q:.3f}) = {info.new_q:.3f}\n\n"
                f"Selección de acción: {info.source}"
            )
        )

    def _do_step(self, greedy_only: bool = False) -> bool:
        state = self.env.agent_pos

        if greedy_only:
            action = self.agent.best_action(state)
            source = "Performance Element (demostración)"
        else:
            action, source = self.agent.choose_action(state)

        result: StepResult = self.env.step(action)
        old_q, max_next_q, target, new_q = self.agent.learn(state, action, result.reward, result.next_state)

        self.step_count += 1
        self.total_reward += result.reward

        self.last_info = StepInfo(
            prev_state=state,
            action=action,
            action_name=GridWorldEnv.ACTION_NAMES[action],
            next_state=result.next_state,
            reward=result.reward,
            done=result.done,
            old_q=old_q,
            max_next_q=max_next_q,
            target=target,
            new_q=new_q,
            source=source,
            event=result.event,
        )

        self.log(
            f"[Critic] s={state} | a={action} ({GridWorldEnv.ACTION_NAMES[action]}) | "
            f"s'={result.next_state} | r={result.reward:.3f} | evento={result.event}"
        )
        self.log(
            f"[Learning Element] Q({state},{action}) {old_q:.3f} -> {new_q:.3f} | "
            f"target={target:.3f} | max_next_q={max_next_q:.3f}"
        )
        self.log(f"[{source}] acción elegida = {action} ({GridWorldEnv.ACTION_NAMES[action]})")
        self.log("-" * 100)

        self._draw_grid()
        self._refresh_panels()

        if result.done:
            outcome = "win" if result.reward > 0 else "trap"
            self.metrics.record(self.total_reward, self.step_count, outcome)
            self.log(f"[Fin Episodio] Resultado={result.event} | pasos={self.step_count} | recompensa total={self.total_reward:.3f}")
            self._finish_episode()
            return True

        if self.step_count >= self.config.max_steps_per_episode:
            self.metrics.record(self.total_reward, self.step_count, "max_steps")
            self.log(f"[Fin Episodio] Límite de pasos alcanzado | recompensa total={self.total_reward:.3f}")
            self._finish_episode()
            return True

        return False

    def _finish_episode(self) -> None:
        self.log("=" * 100)
        self.episode += 1
        self.step_count = 0
        self.total_reward = 0.0
        self.last_info = None
        self.env.reset()
        self._draw_grid()
        self._refresh_panels(initial=True)

    def step_once(self) -> None:
        self._do_step()

    def run_one_episode(self) -> None:
        finished = False
        while not finished:
            finished = self._do_step()

    def train_many(self, episodes: int) -> None:
        wins_before = self.metrics.wins
        traps_before = self.metrics.trap_hits

        for _ in range(episodes):
            self.env.reset()
            episode_reward = 0.0
            steps = 0

            while steps < self.config.max_steps_per_episode:
                state = self.env.agent_pos
                action, _ = self.agent.choose_action(state)
                result = self.env.step(action)
                self.agent.learn(state, action, result.reward, result.next_state)

                episode_reward += result.reward
                steps += 1

                if result.done:
                    outcome = "win" if result.reward > 0 else "trap"
                    self.metrics.record(episode_reward, steps, outcome)
                    break
            else:
                self.metrics.record(episode_reward, steps, "max_steps")

            self.episode += 1

        self.step_count = 0
        self.total_reward = 0.0
        self.last_info = None
        self.env.reset()
        self._draw_grid()
        self._refresh_panels(initial=True)

        self.log(f"[Entrenamiento] Ejecutados {episodes} episodios.")
        self.log(f"[Entrenamiento] Nuevas victorias: {self.metrics.wins - wins_before}")
        self.log(f"[Entrenamiento] Nuevas caídas en obstáculo: {self.metrics.trap_hits - traps_before}")
        self.log("=" * 100)

    def auto_slow_episode(self) -> None:
        self._auto_loop()

    def _auto_loop(self) -> None:
        finished = self._do_step()
        if not finished:
            self.root.after(450, self._auto_loop)

    def show_greedy_demo(self) -> None:
        self.log("[Demostración] Se ejecutará solo la mejor acción conocida por estado.")
        self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_info = None
        self._draw_grid()
        self._refresh_panels(initial=True)
        self._greedy_loop()

    def _greedy_loop(self) -> None:
        finished = self._do_step(greedy_only=True)
        if not finished:
            self.root.after(450, self._greedy_loop)

    def reset_position_only(self) -> None:
        self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.last_info = None
        self._draw_grid()
        self._refresh_panels(initial=True)
        self.log(f"[Reset posición] Nuevo estado inicial = {self.env.agent_pos}")

    def reset_all(self) -> None:
        self.agent.reset_q_table()
        self.metrics.reset()
        self.episode = 1
        self.step_count = 0
        self.total_reward = 0.0
        self.last_info = None
        self.env.reset()
        self.clear_log()
        self._draw_grid()
        self._refresh_panels(initial=True)
        self.log("[Reset total] Tabla Q, métricas y ejecución reiniciadas.")

    def save_q_json(self) -> None:
        filepath = filedialog.asksaveasfilename(
            title="Guardar tabla Q",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not filepath:
            return

        try:
            save_q_table(self.agent.q_table, filepath)
            self.log(f"[Persistencia] Tabla Q guardada en: {filepath}")
            messagebox.showinfo("Guardar tabla Q", f"Tabla Q guardada correctamente.\n\n{filepath}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar la tabla Q.\n\n{exc}")

    def load_q_json(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Cargar tabla Q",
            filetypes=[("JSON", "*.json")],
        )
        if not filepath:
            return

        try:
            self.agent.q_table = load_q_table(filepath)
            self._refresh_panels(initial=True)
            self.log(f"[Persistencia] Tabla Q cargada desde: {filepath}")
            messagebox.showinfo("Cargar tabla Q", f"Tabla Q cargada correctamente.\n\n{filepath}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar la tabla Q.\n\n{exc}")

    def export_convergence_chart(self) -> None:
        filepath = filedialog.asksaveasfilename(
            title="Guardar gráfico de convergencia",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not filepath:
            return

        try:
            self.metrics.save_convergence_chart(filepath)
            self.log(f"[Métricas] Gráfico de convergencia exportado a: {filepath}")
            messagebox.showinfo("Gráfico exportado", f"Gráfico generado correctamente.\n\n{filepath}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo generar el gráfico.\n\n{exc}")
