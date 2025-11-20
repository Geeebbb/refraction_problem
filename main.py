import matplotlib.pyplot as plt
import numpy as np
from refraction2d import Refraction2D, BC, Prism, Impulse, Constants
import matplotlib

matplotlib.use('TkAgg')
import taichi as ti

ti.init(arch=ti.opengl)
from matplotlib.animation import FuncAnimation

# Параметры среды
h = 1.0  # шаг сетки
c = 1.0  # скорость волны
dt = h / (c * 1.5)  # временной шаг
kappa = c * dt / h
n = np.array([1.30, 1.35, 1.40])  # коэффициенты преломления RGB

# Разрешение расчетной области
nx, ny = 900, 600

# Задаём точки измерения амплитуды (нормированные координаты)
point_light = np.array([800 / 900, 500 / 600])
point_dark = np.array([400 / 900, 300 / 600])

# Словарь для записи амплитуд в разных точках
amp_values = {
    'light': [],
    'dark': []
}

# Набор сценариев с разными позициями источника
source_variants = [
    Impulse(freq=400.0, sigma=np.array([0.01, 0.01]), pos=np.array([0.5, 0.5]), angle=0.0),
    Impulse(freq=400.0, sigma=np.array([0.01, 0.01]), pos=np.array([0.2, 0.2]), angle=0.0),
    Impulse(freq=400.0, sigma=np.array([0.01, 0.01]), pos=np.array([0.8, 0.8]), angle=0.0)
]

while True:
    try:
        scenario_id = int(input("введи случай 1,2,3 "))
        if scenario_id in [1, 2, 3]:
            break
    except ValueError:
        print("Некорректный ввод, попробуй ещё раз.")

impulse = source_variants[scenario_id - 1]


# Функция запуска симуляции
def wave_simulation(imp, save_video=False, video_filename="wave_propagation.mp4"):
    """
        Запускает симуляцию распространения волны и создает анимацию.

        Параметры:
            imp (Impulse): Объект Impulse, определяющий параметры источника волны.
            save_video (bool, optional): Если True, сохраняет анимацию в видеофайл
            video_filename (str, optional): Имя файла для сохранения видео
        Возвращает:
            matplotlib.animation.FuncAnimation: Объект анимации Matplotlib.
        """
    ref = Refraction2D(
        nx=nx,
        ny=ny,
        const=Constants(kappa=kappa, n=n),
        impulse=imp,
        prism=None,
        bc=BC.OpenBoundary,
        it=1000,
        refresh=1,
        figsize=(12, 8),
        acc=0.2
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(ref.u[0, 1], cmap='RdBu', vmin=-0.05, vmax=0.05, animated=True, origin='lower',
                    extent=[0, nx, 0, ny])
    plt.colorbar(img, ax=ax, label='Амплитуда волны')

    ax.set_title("Процесс распространения волны", fontsize=16, fontweight='bold',
                 color='#333333')
    ax.set_xlabel("X координата", fontsize=12, color='#555555')
    ax.set_ylabel("Y координата", fontsize=12, color='#555555')
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.tick_params(axis='x', labelsize=10, colors='#666666')
    ax.tick_params(axis='y', labelsize=10, colors='#666666')
    fig.patch.set_facecolor('#F0F0F0')
    ax.set_facecolor('#EAEAEA')

    x_lit = np.clip(int(point_light[0] * nx), 0, nx - 1)
    y_lit = np.clip(int(point_light[1] * ny), 0, ny - 1)
    x_dark = np.clip(int(point_dark[0] * nx), 0, nx - 1)
    y_dark = np.clip(int(point_dark[1] * ny), 0, ny - 1)

    def update_frame(frame):
        """
                Функция обновления кадра для анимации.

                Параметры:
                    frame (int): Текущий номер кадра (временной шаг).

                Возвращает:
                    list: Список Matplotlib Artist объектов, которые были изменены.
                """

        ref.calculate(frame)
        u = ref.u[0, 1]
        img.set_array(u)

        # Запись амплитуды
        amp_values['light'].append(u[y_lit, x_lit])
        amp_values['dark'].append(u[y_dark, x_dark])
        return [img]

    animation = FuncAnimation(fig, update_frame, frames=ref.it, blit=True, interval=20)

    if save_video:
        print(f"Сохранение видео в {video_filename}...")
        animation.save(video_filename, writer='ffmpeg', fps=30, dpi=200)
        print("Видео сохранено.")

    return animation

animation = wave_simulation(impulse, save_video=True, video_filename=f"wave_propagation_scenario_{scenario_id}.mp4")

if isinstance(animation, FuncAnimation):
    plt.show()

plt.figure(figsize=(10, 6))
plt.plot(
    amp_values['light'],
    label='Яркая точка',
    color='#FF5733',
    linewidth=2
)
plt.plot(
    amp_values['dark'],
    label='Теневая точка',
    color='#1F77B4',
    linewidth=2,
    linestyle='--'
)

plt.xlabel('Временной шаг', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.title(f'Динамика амплитуд для случая №{scenario_id}', fontsize=14)

plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

amp_plot_filename = f"amplitude_dynamics_scenario_{scenario_id}.png"
plt.savefig(amp_plot_filename)
print(f"График амплитуд сохранен в {amp_plot_filename}")

plt.show()