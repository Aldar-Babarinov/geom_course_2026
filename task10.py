import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L_sym, R_sym, C_sym, lam = sp.symbols('L R C lambda')
eq = sp.Eq(L_sym * lam**2 + R_sym * lam + 1/C_sym, 0)
eq_sub = eq.subs({L_sym: 1, R_sym: 0.5, C_sym: 0.2})
roots = sp.solve(eq_sub, lam)
print("Корни характеристического уравнения:", roots)

L = 1.0
R = 0.5
C = 0.2
E0 = 1.0
omega = 2.0

def rlc_system(t, y):
    q, I = y
    dqdt = I
    dIdt = (E0 * np.sin(omega * t) - R * I - q / C) / L
    return [dqdt, dIdt]

t_span = (0, 25)
y0 = [0.0, 0.0]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(rlc_system, t_span, y0, t_eval=t_eval, method='RK45')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(sol.t, sol.y[0], 'b-', label='q(t)')
ax1.set_title('Заряд на конденсаторе')
ax1.set_xlabel('Время t, с')
ax1.set_ylabel('Заряд q, Кл')
ax1.grid(True)
ax1.legend()

ax2.plot(sol.t, sol.y[1], 'r-', label='I(t)')
ax2.set_title('Сила тока в контуре')
ax2.set_xlabel('Время t, с')
ax2.set_ylabel('Ток I, А')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()