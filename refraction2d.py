from dataclasses import dataclass, field
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from enum import StrEnum
import taichi as ti
from funcs import (
    triangle_mask,
    calc_impulse,
    rot,
    dirichlet,
    neumann,
    open_boundary,
    propagate,
    accumulate,
    calc_impulse,
    rot,
    mask_biconcave,
    create_penrose_room_mask,
)


class BC(StrEnum):
    Dirichlet = "Dirichlet"
    Neumann = "Neumann"
    OpenBoundary = "OpenBoundary"


@dataclass
class Constants:
    kappa: float
    n: np.ndarray


@dataclass
class Prism:
    pos: np.ndarray
    scale: float


@dataclass
class Impulse:
    freq: float
    sigma: np.ndarray
    pos: np.ndarray
    angle: float


@dataclass
class Colors:
    prism: np.ndarray
    black: np.ndarray = field(default_factory=lambda: np.zeros(3))
    white: np.ndarray = field(default_factory=lambda: np.ones(3))
    red: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    green: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    blue: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    transp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))


class Refraction2D:
    def __init__(
        self,
        const: Constants,
        prism: Prism,
        impulse: Impulse,
        bc: BC = BC.OpenBoundary,
        nx: int = 400,
        ny: int = 400,
        it: int = 1000,
        refresh: int = 5,
        figsize: tuple[float, float] = (12, 6),
        colors: Colors = Colors(prism=np.array([201.0, 228.0, 235.0, 30.0]) / 255.0),
        acc: float = 0.1,
    ):
        """
        :param nx: разрешение по оси X
        :param ny: разрешение по оси Y
        :param kappa: константа волнового уравнения kappa = c * dt / h
        :param bc: граничные условия (см. класс BC)
        :param it: количество итераций
        :param refresh: обновлять рисунок через refresh итераций расчета
        """

        self.nx = nx
        self.ny = ny
        self.bc = bc
        self.const = const
        self.it = it
        self.refresh = refresh
        assert self.refresh > 0
        assert self.it > 0
        assert self.refresh < self.it

        lens_mask = mask_biconcave(
            nx=self.nx,
            ny=self.ny,
            x_center=int(self.nx * 0.6),
            y_center=int(self.ny * 0.5),
            width=120,
            height=300,
            rx=40,
            ry=150
        )
        self.kappa = np.ones((3, ny, nx), dtype=np.float32) * const.kappa
        for c in range(3):
            self.kappa[c][lens_mask] = const.kappa / const.n[c]
        mirror_mask = create_penrose_room_mask(nx=self.nx, ny=self.ny, line_width=3)

        self.u = np.zeros((3, 3, ny, nx), dtype=np.float32)
        self.accum = np.zeros((3, ny, nx), dtype=np.float32)

        self.colors = colors
        self.acc = acc
        fig, ax = plt.subplots(2, 2, figsize=figsize)
        self.fig: Figure = fig
        self.fig.suptitle(f"2D refraction")
        self.ax = ax
        self.progress = trange(self.it)

        self.u += calc_impulse(
            nx, ny, rot(impulse.angle), impulse.pos, impulse.freq, impulse.sigma
        )[None, None]

        self.refraction_shader = RefractionShader(nx, ny, n_colors=3)
        self.refraction_shader.from_numpy(self.u, self.kappa, self.accum, mirror_mask)

    def plot(self):
        """
                Обновляет и отображает графики

                Этот метод очищает существующие подграфики и перерисовывает их,
                визуализируя текущее состояние каждого цветового канала (R, G, B)
                и суммарное накопленное изображение (accum) с использованием
                pcolormesh для создания цветовой карты.
                """
        colors = [self.colors.red, self.colors.green, self.colors.blue]
        titles = ['R(x, y, t)', 'G(x, y, t)', 'B(x, y, t)']

        for layer in range(3):
            arr_ = self.u[layer, 0]
            amax = np.max(np.abs(arr_))
            if amax > 1e-8:
                arr_ = np.clip(arr_ / amax, 0.0, 1.0)
            else:
                arr_ = np.zeros_like(arr_)

            color = colors[layer]
            img = arr_[..., None] * color[None, None, :]
            i = layer // 2
            j = layer % 2
            self.ax[i, j].clear()
            self.ax[i, j].pcolormesh(img)
            self.ax[i, j].set_title(titles[layer])
            self.ax[i, j].set_xlabel("x")
            self.ax[i, j].set_ylabel("y")

        img = np.clip(self.accum.transpose((1, 2, 0)), 0.0, 1.0) ** 2
        self.ax[1, 1].clear()
        self.ax[1, 1].pcolormesh(img)
        self.ax[1, 1].set_title("accum(x, y, t)")
        self.ax[1, 1].set_xlabel("x")
        self.ax[1, 1].set_ylabel("y")

        self.fig.tight_layout()

    def get_animation(self) -> FuncAnimation:
        """
               Создает и возвращает объект анимации Matplotlib для симуляции.

               Возвращает:
                   FuncAnimation: Объект анимации Matplotlib, который может быть использован
                                  для отображения или сохранения видео.
        """
        return FuncAnimation(
            self.fig,
            self.calculate,  # type: ignore
            frames=self.it // self.refresh,
            repeat=False,
        )

    def calculate(self, i: int):
        """
                Выполняет один или несколько шагов симуляции и обновляет данные для отрисовки.

                Этот метод вызывается для каждого кадра анимации. Он продвигает симуляцию
                на `self.refresh` шагов, затем получает обновленные данные волны и
                накопленные данные из шейдера и вызывает метод `plot` для их визуализации.

                Аргументы:
                    i (int): Индекс текущего кадра анимации (не используется напрямую
                             для расчетов шагов симуляции, а для отслеживания прогресса).
                """
        for _ in range(self.refresh):
            self.refraction_shader.step(self.acc, self.const.kappa)
            self.progress.update(1)

        self.u = self.refraction_shader.get_u()
        self.accum = self.refraction_shader.get_accum()
        self.plot()

@ti.data_oriented
class RefractionShader:
    """
        Класс RefractionShader управляет расчетами распространения волн с использованием Taichi.

        Он содержит поля Taichi для хранения состояния волны (u0, u1, u2),
        коэффициентов преломления (kappa), накопленной энергии (accum) и маски
        зеркала (mirror_mask). Он также предоставляет кернелы для выполнения
        различных этапов симуляции, таких как распространение волны,
        обработка граничных условий и накопление энергии.
        """
    def __init__(self, nx, ny, n_colors=3):
        """
                Инициализирует RefractionShader.

                Аргументы:
                    nx (int): Ширина расчетной области.
                    ny (int): Высота расчетной области.
                    n_colors (int, optional): Количество цветовых каналов
                """
        self.nx = nx
        self.ny = ny
        self.n_colors = n_colors

        self.u0 = ti.Vector.field(n_colors, dtype=ti.f32, shape=(ny, nx))
        self.u1 = ti.Vector.field(n_colors, dtype=ti.f32, shape=(ny, nx))
        self.u2 = ti.Vector.field(n_colors, dtype=ti.f32, shape=(ny, nx))

        self.kappa = ti.Vector.field(n_colors, dtype=ti.f32, shape=(ny, nx))
        self.accum = ti.Vector.field(n_colors, dtype=ti.f32, shape=(ny, nx))

        self.mirror_mask = ti.field(dtype=ti.i32, shape=(ny, nx))

    def get_u(self):
        """
                Извлекает данные состояния волны из полей Taichi и преобразует их в массив NumPy.

                Возвращает:
                    np.ndarray: Массив NumPy формы (n_colors, 3, ny, nx),
                                где 3 соответствует u0, u1, u2.
                """
        arr = np.zeros((self.n_colors, 3, self.ny, self.nx), dtype=np.float32)
        arr[:, 0, :, :] = np.moveaxis(self.u0.to_numpy(), -1, 0)
        arr[:, 1, :, :] = np.moveaxis(self.u1.to_numpy(), -1, 0)
        arr[:, 2, :, :] = np.moveaxis(self.u2.to_numpy(), -1, 0)
        return arr

    def get_accum(self):
        """
                Извлекает данные накопленной энергии из поля Taichi и преобразует их в массив NumPy.

                Возвращает:
                    np.ndarray: Массив NumPy формы (n_colors, ny, nx).
                """
        return np.moveaxis(self.accum.to_numpy(), -1, 0)

    def from_numpy(self, u, kappa, accum, mirror_mask):
        """
                Загружает данные из массивов NumPy в поля Taichi.

                Аргументы:
                    u (np.ndarray): Начальное состояние волны. Должен быть формы (n_colors, 3, ny, nx)
                                    или быть адаптированным к ней (см. get_u).
                    kappa (np.ndarray): Массив коэффициентов преломления формы (n_colors, ny, nx).
                    accum (np.ndarray): Начальное состояние накопленной энергии формы (n_colors, ny, nx).
                    mirror_mask (np.ndarray): Булева маска для зеркальных границ формы (ny, nx).
                """
        self.u0.from_numpy(np.moveaxis(u[:, 0], 0, -1))
        self.u1.from_numpy(np.moveaxis(u[:, 1], 0, -1))
        self.u2.from_numpy(np.moveaxis(u[:, 2], 0, -1))
        self.kappa.from_numpy(np.moveaxis(kappa, 0, -1))
        self.accum.from_numpy(np.moveaxis(accum, 0, -1))
        self.mirror_mask.from_numpy(mirror_mask.astype(np.int32))

    @ti.kernel
    def open_boundary_kernel(self):
        """
                Применяет открытые граничные условия к расчетной области.

                Этот кернел корректирует значения волны на границах области,
                чтобы имитировать открытые границы, предотвращая отражения.
                """
        for c in ti.static(range(self.n_colors)):
            for j in range(self.nx):
                self.u0[0, j][c] = (
                    self.u1[1, j][c]
                    + (self.kappa[0, j][c] - 1.0) / (self.kappa[0, j][c] + 1.0)
                    * (self.u0[1, j][c] - self.u1[0, j][c])
                )
                self.u0[self.ny - 1, j][c] = (
                    self.u1[self.ny - 2, j][c]
                    + (self.kappa[self.ny - 1, j][c] - 1.0) / (self.kappa[self.ny - 1, j][c] + 1.0)
                    * (self.u0[self.ny - 2, j][c] - self.u1[self.ny - 1, j][c])
                )
        for c in ti.static(range(self.n_colors)):
            for i in range(self.ny):
                self.u0[i, 0][c] = (
                    self.u1[i, 1][c]
                    + (self.kappa[i, 0][c] - 1.0) / (self.kappa[i, 0][c] + 1.0)
                    * (self.u0[i, 1][c] - self.u1[i, 0][c])
                )
                self.u0[i, self.nx - 1][c] = (
                    self.u1[i, self.nx - 2][c]
                    + (self.kappa[i, self.nx - 1][c] - 1.0) / (self.kappa[i, self.nx - 1][c] + 1.0)
                    * (self.u0[i, self.nx - 2][c] - self.u1[i, self.nx - 1][c])
                )

    @ti.kernel
    def propagate_kernel(self):
        """
                Выполняет шаг распространения волны по всей расчетной области.

                Этот кернел вычисляет новое состояние волны (u0) на основе
                предыдущих состояний (u1, u2) и коэффициента преломления (kappa)
                с использованием метода конечных разностей.
                Также обновляет состояния u2 и u1 для следующего временного шага.
                """
        for c in ti.static(range(self.n_colors)):
            for i, j in ti.ndrange((1, self.ny - 1), (1, self.nx - 1)):
                lap = (
                    self.u1[i + 1, j][c] +
                    self.u1[i - 1, j][c] +
                    self.u1[i, j + 1][c] +
                    self.u1[i, j - 1][c] -
                    4.0 * self.u1[i, j][c]
                )
                self.u0[i, j][c] = (
                    self.kappa[i, j][c] ** 2 * lap
                    + 2.0 * self.u1[i, j][c]
                    - self.u2[i, j][c]
                )
        for i, j in ti.ndrange(self.ny, self.nx):
            self.u2[i, j] = self.u1[i, j]
            self.u1[i, j] = self.u0[i, j]

    @ti.kernel
    def accumulate_kernel(self, coeff: float, kappa_: float):
        """
                Накапливает амплитуду волны для создания кумулятивного изображения.

                Этот кернел добавляет к полю `accum` взвешенное значение текущей
                амплитуды волны. Используется для визуализации пути волны или
                интенсивности за время симуляции.

                Аргументы:
                    coeff (float): Коэффициент для масштабирования накопленной амплитуды.
                    kappa_ (float): Базовый коэффициент преломления для нормализации.
                """
        for c in ti.static(range(self.n_colors)):
            for i, j in ti.ndrange((1, self.ny - 1), (1, self.nx - 1)):
                self.accum[i, j][c] += (
                    coeff * ti.abs(self.u1[i, j][c]) * self.kappa[i, j][c] / kappa_
                )

    @ti.kernel
    def mirror_dirichlet_kernel(self):
        """
                Применяет зеркальные (граничные условия Дирихле) условия.

                В областях, указанных в `mirror_mask`, значения волны устанавливаются в ноль,
                имитируя полное поглощение или идеальное зеркало.
                """
        for i, j in ti.ndrange(self.ny, self.nx):
            if self.mirror_mask[i, j]:
                for c in ti.static(range(self.n_colors)):
                    self.u0[i, j][c] = 0.0
                    self.u1[i, j][c] = 0.0
                    self.u2[i, j][c] = 0.0

    def step(self, acc: float, kappa_: float):
        """
                Выполняет один полный временной шаг симуляции распространения волны.

                Этот метод координирует вызовы всех кернелов Taichi для:
                1. Применения открытых граничных условий.
                2. Вычисления распространения волны.
                3. Применения граничных условий Дирихле для зеркал.
                4. Накопления амплитуды для визуализации.

                Аргументы:
                    acc (float): Коэффициент накопления для accumulate_kernel.
                    kappa_ (float): Базовый коэффициент преломления для accumulate_kernel.
                """
        self.open_boundary_kernel()
        self.propagate_kernel()
        self.mirror_dirichlet_kernel()
        self.accumulate_kernel(acc, kappa_)
