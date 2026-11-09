import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем 'Agg' для сохранения графиков в файлы без окон
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Параметры из таблицы ---
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
g = 9.81
t_span = (0, 20)
N = 1000
t_eval = np.linspace(t_span[0], t_span[1], N)
delta = 1e-6

# --- Функция правых частей (формулы 5.5) ---
def double_pendulum_rhs(t, y):
    theta1, omega1, theta2, omega2 = y
    delta_angles = theta1 - theta2
    
    den1 = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta_angles))
    den2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta_angles))
    
    d_omega1 = (-g * (2*m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2*theta2) 
                - 2 * np.sin(delta_angles) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta_angles))) / den1
    d_omega2 = (2 * np.sin(delta_angles) * (omega1**2 * L1 * (m1 + m2) 
                + g * (m1 + m2) * np.cos(theta1) + omega2**2 * L2 * m2 * np.cos(delta_angles))) / den2
    return [omega1, d_omega1, omega2, d_omega2]

# --- Функция для расчета полной механической энергии ---
def get_energy(y):
    theta1, omega1, theta2, omega2 = y
    T = (0.5 * (m1 + m2) * (L1 * omega1)**2 + 0.5 * m2 * (L2 * omega2)**2 
         + m2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return T + V

# ==========================================
# 1. Расчет случаев А и В
# ==========================================
y0_A = [np.pi/2, 0.0, np.pi/2 + 0.01, 0.0]
y0_B = [np.pi/2, 0.0, np.pi/2 + 0.01 + delta, 0.0]

# Интегрируем с высокой точностью
sol_A = solve_ivp(double_pendulum_rhs, t_span, y0_A, method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-8)
sol_B = solve_ivp(double_pendulum_rhs, t_span, y0_B, method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-8)

# Перевод в декартовы координаты (x2, y2) для нижнего груза
x2_A = L1 * np.sin(sol_A.y[0]) + L2 * np.sin(sol_A.y[2])
y2_A = -L1 * np.cos(sol_A.y[0]) - L2 * np.cos(sol_A.y[2])

x2_B = L1 * np.sin(sol_B.y[0]) + L2 * np.sin(sol_B.y[2])
y2_B = -L1 * np.cos(sol_B.y[0]) - L2 * np.cos(sol_B.y[2])

# ==========================================
# 2. Построение двух траекторий рядом
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

axes[0].plot(x2_A, y2_A, 'b-', lw=1)
axes[0].set_title(r"Случай A: $\theta_2(0) = \pi/2 + 0.01$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True)
axes[0].set_aspect('equal')

axes[1].plot(x2_B, y2_B, 'r-', lw=1)
axes[1].set_title(r"Случай B: $\theta_2(0) = \pi/2 + 0.01 + 10^{-6}$")
axes[1].set_xlabel("x")
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig("task3_2_trajectories.png", dpi=150)
plt.close()

# ==========================================
# 3. Проверка сохранения энергии (1e-8 и 1e-5)
# ==========================================
E_A = np.array([get_energy(sol_A.y[:, i]) for i in range(len(sol_A.t))])
E_0 = E_A[0]

# Относительная ошибка для точности 1e-8
rel_error_8 = np.abs(E_A - E_0) / np.abs(E_0)

# Расчет с точностью 1e-5
sol_A_5 = solve_ivp(double_pendulum_rhs, t_span, y0_A, method='RK45', t_eval=t_eval, rtol=1e-5, atol=1e-5)
E_A_5 = np.array([get_energy(sol_A_5.y[:, i]) for i in range(len(sol_A_5.t))])
rel_error_5 = np.abs(E_A_5 - E_0) / np.abs(E_0)

fig_err, ax_err = plt.subplots(figsize=(8, 5))
ax_err.semilogy(sol_A.t, rel_error_8, 'b-', label='Точность 1e-8')
ax_err.semilogy(sol_A_5.t, rel_error_5, 'r--', label='Точность 1e-5')

ax_err.set_xlabel("Время t, с")
ax_err.set_ylabel(r"$\frac{|E(t) - E(0)|}{|E(0)|}$")
ax_err.set_title("Относительное изменение энергии")
ax_err.grid(True, which="both", alpha=0.3)
ax_err.legend()

plt.tight_layout()
plt.savefig("task3_2_energy_error.png", dpi=150)
plt.close()

print("Задание 3.2 выполнено!")
print("Сохранены файлы: 'task3_2_trajectories.png', 'task3_2_energy_error.png'")