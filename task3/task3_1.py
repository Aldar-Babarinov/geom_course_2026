import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем 'Agg' для сохранения в файлы без открытия окон
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Параметры из таблицы ---
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
g = 9.81

# Начальные условия (случай A)
y0 = [np.pi/2, 0.0, np.pi/2 + 0.01, 0.0]

# Время интегрирования
t_span = (0, 20)
N = 1000
t_eval = np.linspace(t_span[0], t_span[1], N)

# --- 1. Функция правых частей (формулы 5.5) ---
def double_pendulum_rhs(t, y):
    theta1, omega1, theta2, omega2 = y
    
    # Разность углов
    delta = theta1 - theta2
    
    # Вспомогательные величины для сокращения записи
    den1 = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
    den2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
    
    # Уравнения для угловых ускорений (ddot theta)
    d_omega1 = (
        -g * (2*m1 + m2) * np.sin(theta1) 
        - m2 * g * np.sin(theta1 - 2*theta2) 
        - 2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
    ) / den1
    
    d_omega2 = (
        2 * np.sin(delta) * (
            omega1**2 * L1 * (m1 + m2) 
            + g * (m1 + m2) * np.cos(theta1) 
            + omega2**2 * L2 * m2 * np.cos(delta)
        )
    ) / den2
    
    return [omega1, d_omega1, omega2, d_omega2]

# --- 2. Решение задачи Коши ---
print("Идет расчет...")
sol = solve_ivp(
    double_pendulum_rhs, 
    t_span, 
    y0, 
    method='RK45', 
    t_eval=t_eval, 
    rtol=1e-8, 
    atol=1e-8
)

# Извлекаем результаты
theta1 = sol.y[0]
omega1 = sol.y[1]
theta2 = sol.y[2]
omega2 = sol.y[3]
t = sol.t

# --- 3. Перевод в декартовы координаты ---
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# --- 4. Статический график траектории нижнего груза ---
plt.figure(figsize=(8, 8))
plt.plot(x2, y2, 'b-', linewidth=1, alpha=0.7)
plt.plot(x2[0], y2[0], 'go', label='Начало') 
plt.plot(x2[-1], y2[-1], 'ro', label='Конец') 
plt.title("Траектория нижнего груза (x2, y2)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig("trajectory.png", dpi=150) 
plt.close() # Закрываем фигуру, чтобы не забивать память
print("График сохранен в файл: trajectory.png")

# --- 5. Анимация ---
print("Генерация анимации (это может занять несколько секунд)...")
fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
ax_anim.set_xlim(-2.5, 2.5)
ax_anim.set_ylim(-2.5, 2.5)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.set_title("Двойной маятник")

line1, = ax_anim.plot([], [], 'o-', lw=2, color='black')
line_trail, = ax_anim.plot([], [], '-', lw=1, color='blue', alpha=0.5)

def init():
    line1.set_data([], [])
    line_trail.set_data([], [])
    return line1, line_trail

def animate(i):
    this_x = [0, x1[i], x2[i]]
    this_y = [0, y1[i], y2[i]]
    line1.set_data(this_x, this_y)
    line_trail.set_data(x2[:i], y2[:i])
    return line1, line_trail

anim = FuncAnimation(fig_anim, animate, init_func=init, frames=len(t), interval=20, blit=True)

# Сохраняем анимацию в GIF
anim.save("pendulum_animation.gif", writer='pillow', fps=30)
plt.close()
print("Анимация сохранена в файл: pendulum_animation.gif")
print("Готово!")