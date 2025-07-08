from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import sdeint
import sympy as sp
from sympy.physics.units import hbar

from three_variable.coherent_states import (
    action_from_expr,
    extract_action,
    xp_expression_from_alpha,
)
from three_variable.equilibrium_squeeze import get_equilibrium_squeeze_ratio
from three_variable.projected_sse import (
    get_diffusion_term,
    get_environment_derivative,
    get_full_derivative,
    get_system_derivative,
)
from three_variable.simulation import ELENA_LI_CU, ELENA_NA_CU, TOWNSEND_H_RU
from three_variable.symbols import (
    KBT,
    alpha,
    dimensionless_from_full,
    eta_lambda,
    eta_m,
    eta_omega,
    noise,
    zeta,
)

t = sp.Symbol("t", real=True)
Re_xi = sp.Symbol("Re_xi", real=True)
Im_xi = sp.Symbol("Im_xi", real=True)

equilibrium_ratio = get_equilibrium_squeeze_ratio()
low_friction = equilibrium_ratio.lseries(eta_lambda, sp.oo)  # type: ignore sp
R_expr = sum(sp.simplify(e) for e in itertools.islice(low_friction, 1))  # type: ignore sp

PHYSICAL_PARAMS = [
    ("H Ru", TOWNSEND_H_RU.eta_parameters, "C3"),
    ("Li Cu", ELENA_LI_CU.eta_parameters, "C1"),
    ("Na Cu", ELENA_NA_CU.eta_parameters, "C2"),
]

alpha_derivative_deterministic = get_full_derivative("alpha")

alpha_derivative_diffusion = get_diffusion_term()
alpha_derivative_diffusion = sp.factor(
    extract_action(action_from_expr(alpha_derivative_diffusion), "alpha")
)
alpha_derivative_diffusion = sp.simplify(
    dimensionless_from_full(alpha_derivative_diffusion),
    rational=True,
)

alpha_derivative = alpha_derivative_deterministic + alpha_derivative_diffusion

expr_system = get_system_derivative("zeta")
expr_environment = get_environment_derivative("zeta")

zeta_derivative = expr_system + expr_environment


def get_x_p_derivatives(alpha_derivative: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Calculate the derivatives dx/dt and dp/dt from the alpha derivative and equilibrium squeeze ratio,
    and express in (x,p) basis.
    """
    alpha_derivative_conj = sp.conjugate(alpha_derivative)
    a_expec_derivative = (alpha_derivative + alpha_derivative_conj * zeta) / (
        1 - zeta * sp.conjugate(zeta)
    )
    a_dagger_expec_derivative = (
        alpha_derivative_conj + alpha_derivative * sp.conjugate(zeta)
    ) / (1 - zeta * sp.conjugate(zeta))
    dxdt = (a_expec_derivative + a_dagger_expec_derivative) / sp.sqrt(2)
    dpdt = 1j * hbar * (-a_expec_derivative + a_dagger_expec_derivative) / sp.sqrt(2)
    dxdt = sp.simplify(
        xp_expression_from_alpha(dxdt).subs(
            {
                zeta: (1 - R_expr) / (1 + R_expr),
                sp.Symbol("V_1"): 0,
                noise: Re_xi + 1j * Im_xi,
                sp.conjugate(noise): Re_xi - 1j * Im_xi,
            }
        )
    )
    dpdt = sp.simplify(
        xp_expression_from_alpha(dpdt).subs(
            {
                zeta: (1 - R_expr) / (1 + R_expr),
                sp.Symbol("V_1"): 0,
                noise: Re_xi + 1j * Im_xi,
                sp.conjugate(noise): Re_xi - 1j * Im_xi,
            }
        )
    )
    return dxdt, dpdt


dxdt, dpdt = get_x_p_derivatives(alpha_derivative_diffusion)

print("Alpha derivative (deterministic, real part):")
sp.print_latex(dxdt)
print("Alpha derivative (deterministic, imaginary part):")
sp.print_latex(dpdt)

input()
# Substitute physical parameters for numerical evaluation
eta_lambda_value = 0.01
eta_m_value = 1
eta_omega_value = 1
hbar_value = 1
KBT_value = 1

alpha_derivative_deterministic = alpha_derivative_deterministic.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,
    }
)
zeta_derivative = zeta_derivative.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,
    }
)
alpha_derivative_diffusion = alpha_derivative_diffusion.subs(
    {
        sp.Symbol("V_1"): 0,
        eta_lambda: eta_lambda_value,
        eta_m: eta_m_value,
        eta_omega: eta_omega_value,
        KBT: KBT_value,
        hbar: hbar_value,
    }
)

# dX = F dt + G dW
# Lambdify to get NumPy-compatible functions
drift_expr = sp.Matrix([alpha_derivative_deterministic, zeta_derivative])
diff_expr = sp.Matrix([[alpha_derivative_diffusion / noise, 0], [0, 0]])

# lambdify full vector input y = [alpha, zeta]
drift_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), drift_expr, modules="numpy")
diff_func = sp.lambdify((t, sp.Matrix([alpha, zeta])), diff_expr, modules="numpy")


# 5. Wrap for sdeint
def f(y, t):
    return np.array(drift_func(t, y)).astype(np.complex128).flatten()


def G(y, t):
    return np.array(diff_func(t, y)).astype(np.complex128)


# 6. Initial values and time vector
y0 = np.array([1.0 + 0.0j, -2 / 3 + 0.0j])  # alpha and zeta
ts = np.linspace(0, 2, int(eta_m_value * 10000))

print("Starting simulation")
# 7. Solve using Itô interpretation
sol = sdeint.itoint(f, G, y0, ts)

# ts = np.linspace(0, 0.005, 1000)
# sol = sdeint.itoint(f, G, sol[-1, :], ts)

# 8. Extract results
alpha_sol = sol[:, 0]
zeta_sol = sol[:, 1]

# 9. calculate x and p from alpha and zeta
zeta_conj = np.conjugate(zeta_sol)
alpha_conj = np.conjugate(alpha_sol)
a_expec = (alpha_sol + alpha_conj * zeta_sol) / (1 - zeta_sol * zeta_conj)
a_dagger_expec = (alpha_conj + alpha_sol * zeta_conj) / (1 - zeta_sol * zeta_conj)
x_sol = (a_expec + a_dagger_expec) / np.sqrt(2)
p_sol = 1j * (-a_expec + a_dagger_expec) / np.sqrt(2)
dxdt = np.gradient(x_sol, ts)
dpdt = np.gradient(p_sol, ts)

print("Simulation completed")
# print("Final alpha:", alpha_sol[-1])
print("Final zeta:", zeta_sol[-1])
# 9. Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ts, alpha_sol.real, label="Re(α)")
plt.plot(ts, alpha_sol.imag, label="Im(α)")
plt.title("Alpha Evolution")
plt.xlabel("Time")
plt.ylabel("Alpha")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(ts, zeta_sol.real, label="Re(ζ)")
plt.plot(ts, zeta_sol.imag, label="Im(ζ)")
plt.title("Zeta Evolution")
plt.xlabel("Time")
plt.ylabel("Zeta")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("alpha_zeta_evolution.png", dpi=300)
print("Plot saved as alpha_zeta_evolution.png")

# plot the last 100 points x and p
plt.figure(figsize=(12, 6))

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
(line,) = ax1.plot(ts[-100:], (dxdt[-100:].real / 2) / eta_m_value, label="Re(dxdt)")
line.set_marker("x")
ax2.plot(ts[-100:], dpdt[-100:].real, label="Re(dpdt)")
# ax1.title("X Evolution (Last 100 Points)")
# ax1.xlabel("Time")
# ax1.ylabel("X")
# ax1.legend()

ax1.plot(ts[-100:], p_sol[-100:].real, label="Re(p)")
ax2.plot(ts[-100:], dpdt[-100:].imag, label="Im(dpdt)")
# plt.title("P Evolution (Last 100 Points)")
# plt.xlabel("Time")
# plt.ylabel("P")
# plt.legend()
# plt.tight_layout()
# plt.grid()
plt.savefig("x_p_evolution_last_100.png", dpi=300)
print("Plot saved as x_p_evolution_last_100.png")
