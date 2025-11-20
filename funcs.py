import numpy as np
from numba import njit
import taichi as ti

@njit
def propagate(u: np.ndarray, kappa: np.ndarray):
    """
    Один шаг интегрирования уравнений распространения волны по Эйлеру
    """
    u[:, 2, ...] = u[:, 1, ...]
    u[:, 1, ...] = u[:, 0, ...]

    u[:, 0, 1:-1, 1:-1] = (
        kappa[:, 1:-1, 1:-1] ** 2
        * (
            u[:, 1, 0:-2, 1:-1]
            + u[:, 1, 2:, 1:-1]
            + u[:, 1, 1:-1, 0:-2]
            + u[:, 1, 1:-1, 2:]
            - 4 * u[:, 1, 1:-1, 1:-1]
        )
        + 2 * u[:, 1, 1:-1, 1:-1]
        - u[:, 2, 1:-1, 1:-1]
    )


def dirichlet(u: np.ndarray, val: float = 0.0):
    """
    Граничные условия Дирихле
    """
    u[0, :, :, [0, -1]] = val
    u[0, :, [0, -1], :] = val


def neumann(u: np.ndarray):
    """
    Граничные условия Неймана
    """
    u[0, :, :, [0, -1]] = u[0, :, :, [2, -3]]
    u[0, :, [0, -1], :] = u[0, :, [2, -3], :]


def open_boundary(u: np.ndarray, kappa: np.ndarray):
    """
    Граничные условия открытой границы
    """
    # y = 0
    u[:, 0, 0, :] = u[:, 1, 1, :] + (kappa[:, 0, :] - 1) / (kappa[:, 0, :] + 1) * (
        u[:, 0, 1, :] - u[:, 1, 0, :]
    )
    u[:, 0, -1, :] = u[:, 1, -2, :] + (kappa[:, -1, :] - 1) / (kappa[:, -1, :] + 1) * (
        u[:, 0, -2, :] - u[:, 1, -1, :]
    )

    # x = 0
    u[:, 0, :, 0] = u[:, 1, :, 1] + (kappa[:, :, 0] - 1) / (kappa[:, :, 0] + 1) * (
        u[:, 0, :, 1] - u[:, 1, :, 0]
    )
    u[:, 0, :, -1] = u[:, 1, :, -2] + (kappa[:, :, -1] - 1) / (kappa[:, :, -1] + 1) * (
        u[:, 0, :, -2] - u[:, 1, :, -1]
    )


@njit
def wave_impulse(
    point: np.ndarray,  # (n,m,2)
    pos: np.ndarray,
    freq: float,  # float
    sigma: np.ndarray,  # (2,)
):
    """
    Импульс в виде нескольких сконцентрированных волн специальной формы для уменьшения расхождения "пучка".
    Форма - синусоида по направлению x под куполом функции Гаусса в направлениях x и y.
    https://graphtoy.com/?f1(x,t)=exp(-(x%5E2)/2.0/2%5E2)/2&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=cos(20*x)*f1(x,t)&v4=true&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,4.205926793776742

    :param point: точка, в которой необходимо вычислить амплитуду импульса
    :param pos: центр импульса
    :param freq: частота, отвечающая за количество возмущений внутри импульса
    :param sigma: размах купола Гаусса по осям x и y
    :return: амплитуда импульса в точки point
    """
    d = (point - pos) / sigma
    # (d[0]**2 / (sigma[0] ** 2) + d[1]**2 / (sigma[1] ** 2))
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


@njit
def calc_impulse(
    nx: int,
    ny: int,
    s_rot: np.ndarray,
    s_pos: np.ndarray,
    imp_freq: float,
    imp_sigma: np.ndarray,
) -> np.ndarray:
    """
    Расчет импульса возмущений
    """
    res = np.zeros((ny, nx))
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            res[i, j] += wave_impulse(s_rot @ uv, s_pos, imp_freq, imp_sigma)
    return res


@njit
def mix(a: float, b: float, x: float) -> float:
    return a * x + b * (1.0 - x)


@njit
def clamp(x, low, high):
    return np.maximum(np.minimum(x, high), low)


@njit
def smoothstep(edge0, edge1, x):
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit
def rot(a: float):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, -s], [s, c]])


SQRT_3 = 3**0.5
ISQRT_3 = 1 / SQRT_3


@njit
def sd_equilateral_triangle(p):
    """
    SDF равностороннего треугольника
    :param p:
    :return:
    """
    r = np.array([abs(p[0]) - 1.0, p[1] + ISQRT_3])
    if r[0] + SQRT_3 * r[1] > 0.0:
        r = np.array([r[0] - SQRT_3 * r[1], -SQRT_3 * r[0] - r[1]]) * 0.5
    r[0] -= clamp(r[0], -2.0, 0.0)
    return -((r @ r) ** 0.5) * np.sign(r[1])


@njit
def triangle_mask(
    nx: int, ny: int, pos: np.ndarray, scale: float, a: float = 0.01, b: float = 0.0
):
    """
    Расчет треугольной маски размером (ny, nx) с плавным переходом между 0 и 1
    """
    res = np.empty((ny, nx), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            d = sd_equilateral_triangle((uv + pos) * scale)
            res[i, j] = smoothstep(a, b, d)
    return res


@njit
def accumulate(
    accum: np.ndarray, coeff: float, u: np.ndarray, kappa: np.ndarray, kappa_: float
):
    """
    Накопление возмущений, создаваемых волнами
    """
    accum[:, 1:-1, 1:-1] += (
        coeff * np.abs(u[:, 0, 1:-1, 1:-1]) * kappa[:, 1:-1, 1:-1] / (kappa_)
    )


@njit
def line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, width: int):
    """
    Рисует толстую линию, используя алгоритм Брезенхема.

    Эта функция изменяет входную маску, устанавливая пиксели
    вдоль указанного пути линии. Она использует
    вариант алгоритма Брезенхема для рисования линии
    Аргументы:
        mask (np.ndarray): 2D булев массив NumPy, на котором будет нарисована линия.
                           Пиксели, установленные в True, будут представлять линию.
        x0 (int): X-координата начальной точки линии.
        y0 (int): Y-координата начальной точки линии.
        x1 (int): X-координата конечной точки линии.
        y1 (int): Y-координата конечной точки линии.
        width (int): Толщина линии. Линия будет нарисована толщиной 'width' пикселей
                     вокруг центрального пути линии.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        for w in range(-width // 2, width // 2 + 1):
            for h in range(-width // 2, width // 2 + 1):
                px, py = x0 + w, y0 + h

                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
                    mask[py, px] = True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def create_penrose_room_mask(nx: int, ny: int, line_width: int = 2.5) -> np.ndarray:
    """
    Создает 2D булеву маску, представляющую комнату Пенроуза

    Эта функция рисует серию соединенных линий, которые формируют очертания
    комнаты, похожей на треугольник Пенроуза, на булевом холсте.

    Аргументы:
        nx (int): Ширина холста
        ny (int): Высота холста
        line_width (int, optional): Толщина линий, используемых для рисования комнаты.
                                    По умолчанию 3.

    Возвращает:
        np.ndarray: 2D булев массив NumPy формы (ny, nx), где пиксели True
                    представляют нарисованную форму комнаты Пенроуза.
    """
    canvas = np.zeros((ny, nx), dtype=bool)
    top = (nx // 2, ny - 1)
    left_corner = (0, 0)
    right_corner = (nx - 1, 0)
    indent_left = (nx // 4, ny // 2)
    indent_right = (3 * nx // 4, ny // 2)
    bottom_indent = (nx // 2, ny // 3)

    line(canvas, top[0], top[1], indent_left[0], indent_left[1], line_width)
    line(canvas, indent_left[0], indent_left[1], left_corner[0], left_corner[1], line_width)
    line(canvas, left_corner[0], left_corner[1], bottom_indent[0], bottom_indent[1], line_width)
    line(canvas, bottom_indent[0], bottom_indent[1], right_corner[0], right_corner[1], line_width)
    line(canvas, right_corner[0], right_corner[1], indent_right[0], indent_right[1], line_width)
    line(canvas, indent_right[0], indent_right[1], top[0], top[1], line_width)

    return canvas


def mask_biconcave(nx: int, ny: int, x_center: int, y_center: int, width: int, height: int, rx: float,
                   ry: float) -> np.ndarray:
    """
    Создает 2D булеву маску

    Эта функция генерирует маску, определяя прямоугольную область, а затем
    исключая из ее сторон две эллиптические области, в результате чего получается
    форма, которая толще посередине и тоньше по краям

    Аргументы:
        nx (int): Ширина холста
        ny (int): Высота холста
        x_center (int): X-координата центра двояковогнутой формы.
        y_center (int): Y-координата центра двояковогнутой формы.
        width (int): Общая ширина прямоугольной области, определяющей форму.
        height (int): Общая высота прямоугольной области, определяющей форму.
        rx (float): Радиус по X для эллиптических вырезов. Контролирует кривизну
                    вдоль горизонтальной оси.
        ry (float): Радиус по Y для эллиптических вырезов. Контролирует кривизну
                    вдоль вертикальной оси.

    Возвращает:
        np.ndarray: 2D булев массив NumPy формы (ny, nx), где пиксели True
                    представляют двояковогнутую форму.
    """
    canvas = np.zeros((ny, nx), dtype=bool)
    x0 = x_center - width // 2
    x1 = x_center + width // 2
    y0 = y_center - height // 2
    y1 = y_center + height // 2

    for i in range(y0, y1):
        for j in range(x0, x1):
            left_e = ((j - x0) / rx) ** 2 + ((i - y_center) / ry) ** 2 < 1.0
            right_e = ((j - x1) / rx) ** 2 + ((i - y_center) / ry) ** 2 < 1.0
            if not (left_e or right_e):
                if 0 <= i < ny and 0 <= j < nx:
                    canvas[i, j] = True
    return canvas