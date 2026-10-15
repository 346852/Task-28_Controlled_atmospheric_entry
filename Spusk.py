import math
import queue
import threading
import tkinter as tk

from collections import Counter
from dataclasses import dataclass
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# Планета и статическая атмосфера
# ============================================================

R = 6_371_000.0             # м
MU = 3.986004418e14         # м^3/с^2
G0 = 9.80665                # м/с^2

RHO0 = 1.225               # кг/м^3
H_SCALE = 7200.0            # м
ATM_TOP = 150_000.0         # м


@dataclass(frozen=True) # Неизменяемые начальные условия
class Parameters:
    mass: float
    area: float
    cd: float

    entry_height: float
    entry_speed: float
    gamma_deg: float

    n_limit: float

    angle_min: float
    angle_max: float
    angle_step: float

    dt_max: float
    t_max: float

    @property
    def beta(self):
        return self.mass / (self.cd * self.area)


@dataclass
class Result:
    gamma: float
    status: str

    time: np.ndarray
    state: np.ndarray
    speed: np.ndarray
    load: np.ndarray
    height: np.ndarray

    peak_load: float
    peak_time: float
    peak_height: float


STATUS_TEXT = {
    "success": "Допустимый спуск",
    "overload": "Превышена допустимая перегрузка",
    "exit": "Повторное пересечение границы входа наружу",
    "timeout": "Достижение поверхности не подтверждено за заданное время",
}


def density(height):
    """Статическая экспоненциальная атмосфера."""
    if height > ATM_TOP:
        return 0.0
    return RHO0 * math.exp(-max(0.0, height) / H_SCALE)


def observables(state, p):
    x, y, vx, vy = state
    height = math.hypot(x, y) - R
    speed = math.hypot(vx, vy)

    drag_acc = density(height) * speed * speed / (2.0 * p.beta)
    load = drag_acc / G0

    return height, speed, load


def rhs(state, p):
    x, y, vx, vy = state

    radius = math.hypot(x, y)
    speed = math.hypot(vx, vy)

    gravity_factor = -MU / radius**3
    drag_factor = density(radius - R) * speed / (2.0 * p.beta)

    return (
        vx,
        vy,
        gravity_factor * x - drag_factor * vx,
        gravity_factor * y - drag_factor * vy,
    )


def shifted(state, derivative, scale):
    return tuple(
        state[i] + scale * derivative[i]
        for i in range(4)
    )


def rk4_step(state, dt, p):
    k1 = rhs(state, p)
    k2 = rhs(shifted(state, k1, dt / 2.0), p)
    k3 = rhs(shifted(state, k2, dt / 2.0), p)
    k4 = rhs(shifted(state, k3, dt), p)

    return tuple(
        state[i] + dt * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]
        ) / 6.0
        for i in range(4)
    )


def crossing_fraction(start, end, radius, outward):
    """
    Локализация пересечения окружности на отрезке между
    соседними состояниями. Для точности уменьшайте dt_max.
    """
    left, right = 0.0, 1.0

    for _ in range(35):
        middle = (left + right) / 2.0

        x = start[0] + middle * (end[0] - start[0])
        y = start[1] + middle * (end[1] - start[1])
        inside = math.hypot(x, y) < radius

        # Наружу: сначала внутри, затем снаружи.
        # К поверхности: сначала снаружи, затем внутри.
        if inside == outward:
            left = middle
        else:
            right = middle

    return (left + right) / 2.0


def simulate(p, gamma=None, keep_history=True, stop_on_overload=False):
    """
    RK4 с ограничением шага по характерным временам движения.

    Для исследования коридора историю можно не сохранять,
    а расчет останавливать сразу после превышения перегрузки.
    """
    if gamma is None:
        gamma = p.gamma_deg

    angle = math.radians(gamma)
    entry_radius = R + p.entry_height

    state = (
        entry_radius,
        0.0,
        p.entry_speed * math.sin(angle),
        p.entry_speed * math.cos(angle),
    )

    time = 0.0
    times = [time]
    states = [state]

    height, speed, load = observables(state, p)

    peak_load = load
    peak_time = 0.0
    peak_height = height

    next_record = 1.0
    event = None

    if stop_on_overload and load > p.n_limit:
        event = "overload"

    while time < p.t_max and event is None:
        height, speed, load = observables(state, p)

        radius = math.hypot(state[0], state[1])
        radial_speed = (
            state[0] * state[2] + state[1] * state[3]
        ) / radius

        drag_acc = load * G0

        # Ограничение изменения плотности и скорости за шаг.
        dt_density = 0.05 * H_SCALE / max(abs(radial_speed), 1.0)
        dt_drag = 0.05 * max(speed, 1.0) / max(drag_acc, 1e-12)

        dt = min(
            p.dt_max,
            dt_density,
            dt_drag,
            p.t_max - time,
        )

        new_state = rk4_step(state, dt, p)
        new_radius = math.hypot(new_state[0], new_state[1])

        if new_radius <= R:
            fraction = crossing_fraction(
                state, new_state, R, outward=False
            )
            new_state = tuple(
                state[i] + fraction * (new_state[i] - state[i])
                for i in range(4)
            )
            dt *= fraction
            event = "ground"

        elif radius < entry_radius and new_radius >= entry_radius:
            fraction = crossing_fraction(
                state, new_state, entry_radius, outward=True
            )
            new_state = tuple(
                state[i] + fraction * (new_state[i] - state[i])
                for i in range(4)
            )
            dt *= fraction
            event = "exit"

        time += dt
        state = new_state

        height, speed, load = observables(state, p)

        if load > peak_load:
            peak_load = load
            peak_time = time
            peak_height = height

        if (
            stop_on_overload
            and peak_load > p.n_limit
            and event is None
        ):
            event = "overload"

        # История нужна только для отображения.
        # Максимум перегрузки проверяется на каждом шаге.
        if keep_history and (time >= next_record or event is not None):
            times.append(time)
            states.append(state)
            next_record = time + 1.0

    if times[-1] != time:
        times.append(time)
        states.append(state)

    if event == "ground":
        status = (
            "success" if peak_load <= p.n_limit
            else "overload"
        )
    elif event is not None:
        status = event
    else:
        status = "timeout"

    times_array = np.asarray(times, dtype=float)
    states_array = np.asarray(states, dtype=float)

    heights = np.hypot(states_array[:, 0], states_array[:, 1]) - R
    speeds = np.hypot(states_array[:, 2], states_array[:, 3])

    loads = np.asarray([
        observables(s, p)[2]
        for s in states_array
    ])

    return Result(
        gamma=gamma,
        status=status,
        time=times_array,
        state=states_array,
        speed=speeds,
        load=loads,
        height=heights,
        peak_load=peak_load,
        peak_time=peak_time,
        peak_height=peak_height,
    )


# ============================================================
# Орбита, соответствующая состоянию в точке входа
# ============================================================

def initial_orbit(p):
    gamma = math.radians(p.gamma_deg)
    radius = R + p.entry_height

    vx = p.entry_speed * math.sin(gamma)
    vy = p.entry_speed * math.cos(gamma)

    angular_momentum = radius * vy
    energy = p.entry_speed**2 / 2.0 - MU / radius

    # Вектор эксцентриситета.
    ex = vy * angular_momentum / MU - 1.0
    ey = -vx * angular_momentum / MU

    eccentricity = math.hypot(ex, ey)
    omega = math.atan2(ey, ex)

    semilatus_rectum = angular_momentum**2 / MU

    anomaly = np.linspace(-math.pi, math.pi, 1600)
    denominator = 1.0 + eccentricity * np.cos(anomaly)

    orbit_radius = np.full_like(anomaly, np.nan)

    mask = denominator > 1e-8
    orbit_radius[mask] = semilatus_rectum / denominator[mask]

    # Не показываем часть математической орбиты внутри планеты
    # и слишком удаленные участки.
    visible = (
        (orbit_radius >= R)
        & (orbit_radius <= 3.0 * R)
    )
    orbit_radius[~visible] = np.nan

    theta = anomaly + omega

    ox = orbit_radius * np.cos(theta) / 1000.0
    oy = orbit_radius * np.sin(theta) / 1000.0

    pericenter_height = semilatus_rectum / (1.0 + eccentricity) - R

    if energy < 0.0:
        semimajor_axis = -MU / (2.0 * energy)
        apocenter_height = semimajor_axis * (1.0 + eccentricity) - R

        description = (
            f"Начальная орбита: эллиптическая, e={eccentricity:.4f}\n"
            f"Перицентр: {pericenter_height / 1000:.1f} км; "
            f"апоцентр: {apocenter_height / 1000:.1f} км.\n"
        )
    else:
        description = (
            f"Начальная траектория: незамкнутая, e={eccentricity:.4f}\n"
            f"Перицентр: {pericenter_height / 1000:.1f} км.\n"
        )

    if pericenter_height < 0:
        description += (
            "Отрицательная высота перицентра означает, что "
            "вакуумная орбита пересекает планету.\n"
        )

    return ox, oy, description


# ============================================================
# Поиск допустимого коридора
# ============================================================

def find_corridor(p, progress):
    """
    Сначала сканируем заданную сетку углов.
    Затем уточняем обнаруженные переходы допустим/недопустим.

    Не предполагаем заранее, что допустимый интервал единственный.
    """
    cache = {}

    def evaluate(angle):
        key = round(float(angle), 12)

        if key not in cache:
            cache[key] = simulate(
                p,
                gamma=float(angle),
                keep_history=False,
                stop_on_overload=True,
            )

        return cache[key]

    count = math.ceil(
        (p.angle_max - p.angle_min) / p.angle_step
    )
    angles = np.linspace(p.angle_min, p.angle_max, count + 1)

    accepted = []
    scan_statuses = []

    for index, angle in enumerate(angles):
        result = evaluate(angle)
        accepted.append(result.status == "success")
        scan_statuses.append(result.status)

        progress(
            f"Коридор: {index + 1}/{len(angles)}, "
            f"γ = {angle:.3f}°"
        )

    tolerance = min(0.01, p.angle_step / 10.0)

    def refine(a, b, a_is_valid):
        while b - a > tolerance:
            middle = (a + b) / 2.0
            middle_is_valid = evaluate(middle).status == "success"

            if middle_is_valid == a_is_valid:
                a = middle
            else:
                b = middle

        # Возвращаем точку с допустимой стороны перехода.
        return a if a_is_valid else b

    intervals = []
    index = 0

    while index < len(angles):
        if not accepted[index]:
            index += 1
            continue

        start = index

        while index + 1 < len(angles) and accepted[index + 1]:
            index += 1

        end = index

        left_clipped = start == 0
        right_clipped = end == len(angles) - 1

        left = float(angles[start])
        right = float(angles[end])

        if not left_clipped:
            left = refine(
                float(angles[start - 1]),
                float(angles[start]),
                False,
            )

        if not right_clipped:
            right = refine(
                float(angles[end]),
                float(angles[end + 1]),
                True,
            )

        intervals.append((
            left, right, left_clipped, right_clipped
        ))
        index += 1

    return (
        intervals,
        Counter(scan_statuses),
        len(cache),
        tolerance,
    )


# ============================================================
# Графический интерфейс
# ============================================================

class EntrySimulator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Баллистический вход в атмосферу")
        self.geometry("1380x900")
        self.minsize(1100, 740)

        self.messages = queue.Queue()
        self.animation_id = None
        self.variables = {}

        self.build_interface()
        self.after(100, self.poll_messages)

    def build_interface(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        charts = ttk.Frame(self, padding=5)
        charts.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fields = [
            ("mass", "Масса, кг", "2000"),
            ("area", "Площадь, м²", "10"),
            ("cd", "Коэффициент Cd", "1.2"),
            ("entry_height", "Высота входа, км", "120"),
            ("entry_speed", "Скорость входа, м/с", "7800"),
            ("gamma_deg", "Угол входа, град", "-5"),
            ("n_limit", "Предельная перегрузка, g", "8"),
            ("angle_min", "Нижний угол поиска, град", "-15"),
            ("angle_max", "Верхний угол поиска, град", "-0.2"),
            ("angle_step", "Шаг поиска, град", "0.25"),
            ("dt_max", "Макс. шаг интегрирования, с", "0.5"),
            ("t_max", "Предельное время, с", "4000"),
        ]

        ttk.Label(
            controls,
            text="Параметры спуска",
            font=("", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(0, 10))

        for row, (key, label, default) in enumerate(fields, start=1):
            ttk.Label(controls, text=label).grid(
                row=row, column=0, sticky="w", pady=3
            )

            variable = tk.StringVar(value=default)
            self.variables[key] = variable

            ttk.Entry(
                controls, textvariable=variable, width=12
            ).grid(row=row, column=1, padx=(8, 0), pady=3)

        row = len(fields) + 1

        self.run_button = ttk.Button(
            controls,
            text="Моделировать спуск",
            command=lambda: self.start_job("simulation"),
        )
        self.run_button.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(12, 5)
        )

        self.corridor_button = ttk.Button(
            controls,
            text="Рассчитать коридор",
            command=lambda: self.start_job("corridor"),
        )
        self.corridor_button.grid(
            row=row + 1, column=0, columnspan=2, sticky="ew", pady=5
        )

        ttk.Label(
            controls, text="Скорость анимации, модельных с/кадр"
        ).grid(
            row=row + 2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        self.animation_speed = tk.DoubleVar(value=10.0)

        ttk.Scale(
            controls,
            from_=1.0,
            to=100.0,
            variable=self.animation_speed,
        ).grid(
            row=row + 3, column=0, columnspan=2, sticky="ew"
        )

        self.status_var = tk.StringVar(value="Готово")

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=310,
        ).grid(
            row=row + 4, column=0, columnspan=2,
            sticky="w", pady=10,
        )

        self.report = ScrolledText(
            controls,
            width=43,
            height=20,
            wrap=tk.WORD,
            font=("", 10),
        )
        self.report.grid(
            row=row + 5, column=0, columnspan=2, sticky="nsew"
        )

        controls.rowconfigure(row + 5, weight=1)

        self.figure = Figure(figsize=(9, 8), dpi=100)
        grid = self.figure.add_gridspec(2, 2)

        self.ax_trajectory = self.figure.add_subplot(grid[0, :])
        self.ax_speed = self.figure.add_subplot(grid[1, 0])
        self.ax_load = self.figure.add_subplot(grid[1, 1])

        self.figure.subplots_adjust(
            left=0.09, right=0.97,
            bottom=0.09, top=0.95,
            hspace=0.37, wspace=0.30,
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=charts)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax_trajectory.set_title("Траектория спуска")
        self.ax_speed.set_title("Скорость")
        self.ax_load.set_title("Аэродинамическая перегрузка")
        self.canvas.draw()

        self.write_report(
            "Планета: Земля.\n"
            "Атмосфера: неподвижная, экспоненциальная.\n\n"
            "Угол входа отрицателен и отсчитывается от "
            "местной горизонтали.\n\n"
            "Допустимый спуск: достижение поверхности за "
            "заданное время без превышения перегрузки и "
            "без выхода наружу через высоту входа.\n\n"
            "Безопасность посадки и тепловая защита "
            "в данной модели не оцениваются."
        )

    def write_report(self, text):
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, text)

    def read_parameters(self):
        values = {
            key: float(var.get().replace(",", "."))
            for key, var in self.variables.items()
        }

        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Все значения должны быть конечными числами.")

        values["entry_height"] *= 1000.0
        p = Parameters(**values)

        if min(p.mass, p.area, p.cd, p.n_limit) <= 0:
            raise ValueError(
                "Масса, площадь, Cd и предельная перегрузка должны быть > 0."
            )

        if not 60_000 <= p.entry_height <= ATM_TOP:
            raise ValueError("Высота входа должна быть от 60 до 150 км.")

        if not 100 <= p.entry_speed <= 16_000:
            raise ValueError("Скорость должна быть от 100 до 16000 м/с.")

        if not -89.0 <= p.gamma_deg < 0.0:
            raise ValueError("Угол входа должен быть в диапазоне [-89; 0).")

        if not -89.0 <= p.angle_min < p.angle_max < 0.0:
            raise ValueError(
                "Для поиска требуется: -89 ≤ нижний угол < верхний угол < 0."
            )

        if p.angle_step <= 0:
            raise ValueError("Шаг поиска должен быть положительным.")

        if math.ceil(
            (p.angle_max - p.angle_min) / p.angle_step
        ) > 2000:
            raise ValueError("Слишком густая сетка: максимум 2000 интервалов.")

        if not 0.01 <= p.dt_max <= 2.0:
            raise ValueError("Шаг интегрирования должен быть от 0.01 до 2 с.")

        if not 100 <= p.t_max <= 20_000:
            raise ValueError("Предельное время должно быть от 100 до 20000 с.")

        return p

    def stop_animation(self):
        if self.animation_id is not None:
            self.after_cancel(self.animation_id)
            self.animation_id = None

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.run_button.configure(state=state)
        self.corridor_button.configure(state=state)

    def start_job(self, kind):
        try:
            parameters = self.read_parameters()
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Ошибка параметров", str(error))
            return

        self.stop_animation()
        self.set_busy(True)
        self.status_var.set("Выполняется расчет...")

        def worker():
            try:
                if kind == "simulation":
                    result = simulate(parameters)
                else:
                    result = find_corridor(
                        parameters,
                        lambda text: self.messages.put(("progress", text)),
                    )

                self.messages.put(("done", kind, parameters, result))

            except Exception as error:
                self.messages.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def poll_messages(self):
        try:
            while True:
                message = self.messages.get_nowait()

                if message[0] == "progress":
                    self.status_var.set(message[1])

                elif message[0] == "error":
                    self.set_busy(False)
                    self.status_var.set("Ошибка расчета")
                    messagebox.showerror("Ошибка", message[1])

                elif message[0] == "done":
                    _, kind, parameters, result = message
                    self.set_busy(False)

                    if kind == "simulation":
                        self.show_simulation(parameters, result)
                    else:
                        self.show_corridor(parameters, result)

        except queue.Empty:
            pass

        self.after(100, self.poll_messages)

    def show_simulation(self, p, result):
        ax = self.ax_trajectory
        av = self.ax_speed
        an = self.ax_load

        for axis in (ax, av, an):
            axis.clear()

        states = result.state

        # Координаты графика повернуты для удобства:
        # горизонтальная ось — y, вертикальная — x-R.
        horizontal = states[:, 1] / 1000.0
        vertical = states[:, 0] / 1000.0 - R / 1000.0

        theta = np.linspace(-math.pi, math.pi, 2400)

        planet_horizontal = R * np.sin(theta) / 1000.0
        planet_vertical = R * np.cos(theta) / 1000.0 - R / 1000.0

        ax.fill(
            planet_horizontal, planet_vertical,
            color="#dce6d5", label="Планета",
        )
        ax.plot(planet_horizontal, planet_vertical, color="gray", lw=1)

        boundary_radius = R + p.entry_height

        ax.plot(
            boundary_radius * np.sin(theta) / 1000.0,
            boundary_radius * np.cos(theta) / 1000.0 - R / 1000.0,
            "--", color="skyblue", lw=1,
            label="Граница входа",
        )

        orbit_x, orbit_y, orbit_description = initial_orbit(p)

        ax.plot(
            orbit_y,
            orbit_x - R / 1000.0,
            "--", color="gray", lw=1.2,
            label="Начальная орбита без атмосферы",
        )

        ax.plot(
            horizontal[0], vertical[0],
            "go", label="Точка входа",
        )

        trajectory_line, = ax.plot(
            [], [], color="tab:blue", lw=2, label="Спуск"
        )
        vehicle_point, = ax.plot([], [], "ro", markersize=6)

        ax.set_title("Траектория в плоскости движения")
        ax.set_xlabel("y, км")
        ax.set_ylabel("x − R, км")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        span_x = max(float(np.ptp(horizontal)), 100.0)
        span_y = max(float(np.ptp(vertical)), 100.0)

        ax.set_xlim(
            float(horizontal.min()) - 0.08 * span_x,
            float(horizontal.max()) + 0.08 * span_x,
        )
        ax.set_ylim(
            float(vertical.min()) - 0.12 * span_y,
            float(vertical.max()) + 0.12 * span_y,
        )

        # Масштабы по осям различаются: это удобно для
        # наблюдения высоты над криволинейной поверхностью.
        av.set_title("Скорость")
        av.set_xlabel("Время, с")
        av.set_ylabel("Скорость, км/с")
        av.grid(alpha=0.3)

        an.set_title("Аэродинамическая перегрузка")
        an.set_xlabel("Время, с")
        an.set_ylabel("Перегрузка, g")
        an.grid(alpha=0.3)
        an.axhline(
            p.n_limit, color="red", ls="--",
            label=f"Предел: {p.n_limit:g} g",
        )
        an.legend(fontsize=8)

        speed_line, = av.plot([], [], color="tab:blue")
        load_line, = an.plot([], [], color="tab:orange")

        final_time = float(result.time[-1])

        av.set_xlim(0, max(final_time, 1.0))
        an.set_xlim(0, max(final_time, 1.0))

        av.set_ylim(0, max(float(result.speed.max()) / 1000 * 1.1, 0.1))
        an.set_ylim(0, max(result.peak_load, p.n_limit) * 1.15)

        angles = np.unwrap(np.arctan2(states[:, 1], states[:, 0]))
        ground_range = abs(angles[-1] - angles[0]) * R / 1000.0

        final_label = (
            "Скорость у поверхности"
            if abs(result.height[-1]) < 1.0
            else "Скорость в конце расчета"
        )

        text = (
            f"{STATUS_TEXT[result.status]}\n\n"
            f"Угол входа: {p.gamma_deg:.3f}°\n"
            f"Скорость входа: {p.entry_speed:.1f} м/с\n"
            f"Высота входа: {p.entry_height / 1000:.1f} км\n"
            f"Баллистический коэффициент: {p.beta:.2f} кг/м²\n\n"
            f"Время: {final_time:.1f} с\n"
            f"Конечная высота: {max(0, result.height[-1]) / 1000:.3f} км\n"
            f"{final_label}: {result.speed[-1]:.1f} м/с\n"
            f"Дальность по поверхности: {ground_range:.1f} км\n\n"
            f"Максимальная перегрузка: {result.peak_load:.3f} g\n"
            f"Время максимума: {result.peak_time:.1f} с\n"
            f"Высота максимума: {max(0, result.peak_height) / 1000:.2f} км\n"
            f"Допустимая перегрузка: {p.n_limit:.2f} g\n\n"
            + orbit_description
            + "\nКритерий: поверхность достигнута за отведенное "
              "время, перегрузка не превышена, выхода через "
              "границу входа наружу нет.\n\n"
              "Нагрев и безопасность посадки не оцениваются.\n"
              "Масштабы осей траектории различаются."
        )

        if result.status in ("exit", "timeout"):
            text += (
                "\nОграничение перегрузки: "
                + (
                    "соблюдено на рассчитанном участке."
                    if result.peak_load <= p.n_limit
                    else "также нарушено."
                )
            )

        self.write_report(text)

        playback_time = 0.0

        def animate():
            nonlocal playback_time

            index = int(np.searchsorted(
                result.time, playback_time, side="right"
            ))
            index = max(1, min(index, len(result.time)))

            trajectory_line.set_data(
                horizontal[:index], vertical[:index]
            )
            vehicle_point.set_data(
                [horizontal[index - 1]], [vertical[index - 1]]
            )

            speed_line.set_data(
                result.time[:index], result.speed[:index] / 1000.0
            )
            load_line.set_data(
                result.time[:index], result.load[:index]
            )

            self.status_var.set(
                f"Анимация: t = {result.time[index - 1]:.1f} с; "
                f"h = {max(0, result.height[index - 1]) / 1000:.1f} км"
            )

            self.canvas.draw_idle()

            if index < len(result.time):
                playback_time += self.animation_speed.get()
                self.animation_id = self.after(35, animate)
            else:
                self.animation_id = None
                self.status_var.set(STATUS_TEXT[result.status])

        animate()

    def show_corridor(self, p, data):
        intervals, counts, evaluations, tolerance = data

        text = (
            "РАСЧЕТ ДОПУСТИМОГО КОРИДОРА\n\n"
            f"Высота входа: {p.entry_height / 1000:.1f} км\n"
            f"Скорость входа: {p.entry_speed:.1f} м/с\n"
            f"Масса: {p.mass:g} кг; площадь: {p.area:g} м²\n"
            f"Cd: {p.cd:g}; β: {p.beta:.2f} кг/м²\n"
            f"Предел перегрузки: {p.n_limit:g} g\n"
            f"Предельное время: {p.t_max:g} с\n\n"
            f"Область поиска: [{p.angle_min:g}; {p.angle_max:g}]°\n"
            f"Шаг сетки не более: {p.angle_step:g}°\n"
            f"Уточнение переходов до: {tolerance:.4f}°\n\n"
        )

        if intervals:
            text += "Обнаруженные допустимые интервалы:\n"

            for number, (left, right, clip_left, clip_right) in enumerate(
                intervals, start=1
            ):
                text += (
                    f"{number}. [{left:.4f}; {right:.4f}]°\n"
                    f"   Ширина: {right - left:.4f}°\n"
                )

                if clip_left or clip_right:
                    text += (
                        "   Интервал касается границы области поиска; "
                        "полная граница коридора с этой стороны "
                        "не определена.\n"
                    )
        else:
            text += (
                "Допустимые углы на заданной сетке не обнаружены.\n"
            )

        text += "\nРезультаты первичного сканирования:\n"

        for status, count in counts.items():
            text += f"• {STATUS_TEXT[status]}: {count}\n"

        text += (
            f"\nВсего расчетов с уточнениями: {evaluations}\n\n"
            "Коридор относится к однопроходному спуску. "
            "Повторные входы после выхода через границу "
            "входа не рассчитываются.\n\n"
            "Сетка может пропустить узкий допустимый или "
            "недопустимый участок. Проверьте результат "
            "при меньшем шаге поиска.\n\n"
            "Точность уточнения по углу не равна физической "
            "точности модели. Для проверки численной "
            "сходимости уменьшите шаг интегрирования вдвое.\n\n"
            "Для анимации выберите угол из интервала и "
            "нажмите «Моделировать спуск»."
        )

        self.write_report(text)
        self.status_var.set("Расчет коридора завершен")


if __name__ == "__main__":
    app = EntrySimulator()
    app.mainloop()
