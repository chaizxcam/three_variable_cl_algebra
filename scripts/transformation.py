from __future__ import annotations

import sympy as sp

from three_variable.symbols import (
    a_dagger_expr,
    a_expr,
    zeta,
)

# analyze transformed ladder operators
A = (
    1 / sqrt(1 - zeta * sp.conjugate(zeta)) * (a_expr + zeta * a_dagger_expr)
)  # transformed ladder operator A
# define the constants
s = sp.Symbol("s", real=True)  # sin
c = sp.Symbol("c", real=True)  # cos
r = sp.Symbol("r", real=True)  # r = |zeta|
F = sp.Symbol(
    "F", complex=True
)  # system derivative of alpha + part of the environment derivative
G = sp.Symbol("G", complex=True)  # part of environment derivative of alpha
Re = sp.re(F)
Im = sp.im(F)
E = sp.Symbol("E", real=True)  # E = exp(2*r)
R = sp.Symbol("R", complex=True)  # R = (1 - zeta) / (1 + zeta)

# define the transformation matrices
Rotation = sp.Matrix([[c, s], [-s, c]])
Rotation_inv = sp.Matrix([[c, -s], [s, c]])
P = sp.Matrix([[sp.exp(-r), 0], [0, sp.exp(r)]])

K = sp.Matrix([[1, 1], [-sp.I, sp.I]]) / sp.sqrt(
    2
)  # K_alpha_to_x is the transformation matrix: x = K_alpha_to_x * alpha

Q = sp.Matrix(
    [
        [1, zeta],
        [sp.conjugate(zeta), 1],
    ]
) / (
    1 - sp.Abs(zeta) ** 2
)  # K_alpha_to_a is the transformation matrix: a = K_alpha_to_a * alpha

# S_a is the squeezing matrix in the ladder operator basis


def get_s_a() -> sp.Matrix:
    return (
        sp.Matrix(
            [
                [E + 1 / E, -(E - 1 / E) * (c + sp.I * s)],
                [-(E - 1 / E) * (c - sp.I * s), E + 1 / E],
            ]
        )
        / 2
    )


S_a = get_s_a()


# S_x is the squeezing matrix in the (x, p) basis
def get_s_x() -> sp.Matrix:
    S_a = get_s_a()
    S_x = K * S_a * K.inv()
    S_x = S_x.applyfunc(
        lambda expr: expr.subs({c**2: 1 - s**2, zeta: (1 - R) / (1 + R)})
    )
    return sp.simplify(S_x)


S_x = get_s_x()


# # dynamic matrix in alpha basis
# M_alpha = sp.Matrix(
#     [
#         [F, G],
#         [sp.conjugate(G), sp.conjugate(F)],
#     ]
# )

# # apply squeezing transformation and change to (x, p) basis
# M_x = K * S_a * Q * M_alpha * Q.inv() * S_a.inv() * K.inv()
# # un-squeezed
# # M_x_2 = K * Q * M_alpha * Q.inv() * K.inv()

# # simplify the dynamic matrix in (x, p) basis
# M_x = M_x.applyfunc(lambda expr: expr.subs({c**2: 1 - s**2, zeta: (1 - R) / (1 + R)}))
# # sub s=0 in M_x
# # M_x = M_x.applyfunc(lambda expr: expr.subs({s: 0, c: 1}))
# M_x = sp.simplify(M_x)
# # M_x_2 = M_x_2.applyfunc(
# #     lambda expr: expr.subs({c**2: 1 - s**2, zeta: (1 - R) / (1 + R)})
# # )
# # M_x_2 = sp.simplify(M_x_2)

# # print the results
# sp.print_latex(M_x)
# sp.print_latex(M_x_2)
