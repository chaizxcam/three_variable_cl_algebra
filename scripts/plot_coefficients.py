from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coeffs = pd.read_csv("numerical_derivative_scan_params.csv")
coeffs = coeffs.map(lambda x: complex(x) if isinstance(x, str) and "j" in x else x)


# Plot the coefficients of dx/dt and dp/dt against eta_lambda
def plot_coefficients(
    coeffs: pd.DataFrame, param: str, coeffs_to_plot: list[str]
) -> None:
    derivative = "dx/dt" if "dxdt" in coeffs_to_plot[0] else "dp/dt"

    # -------- Real part plot --------
    fig_real, ax_real = plt.subplots(figsize=(10, 6))
    for coeff in coeffs_to_plot:
        ax_real.plot(coeffs[param], coeffs[coeff].apply(np.real), label=f"Re({coeff})")
    ax_real.set_xscale("log")
    ax_real.set_xlabel(param)
    ax_real.set_yscale("log")
    ax_real.set_ylabel(f"Real part of coefficients of {derivative}")
    ax_real.set_title(f"Real Coefficients of {derivative} vs {param}")
    ax_real.legend()
    plt.tight_layout()
    plt.savefig(
        f"coefficients_{derivative.replace('/', '_')}_real_vs_{param}.png", dpi=300
    )
    plt.close(fig_real)

    # -------- Imaginary part plot --------
    fig_imag, ax_imag = plt.subplots(figsize=(10, 6))
    for coeff in coeffs_to_plot:
        ax_imag.plot(coeffs[param], coeffs[coeff].apply(np.imag), label=f"Im({coeff})")
    ax_imag.set_xscale("log")
    ax_imag.set_xlabel(param)
    ax_imag.set_yscale("log")
    ax_imag.set_ylabel(f"Imaginary part of coefficients of {derivative}")
    ax_imag.set_title(f"Imaginary Coefficients of {derivative} vs {param}")
    ax_imag.legend()
    plt.tight_layout()
    plt.savefig(
        f"coefficients_{derivative.replace('/', '_')}_imag_vs_{param}.png", dpi=300
    )
    plt.close(fig_imag)


# Plot noise coefficients for dx/dt
plot_coefficients(
    coeffs,
    "eta_lambda",
    ["dxdt_coeff_x", "dxdt_coeff_p"],
)

# Plot noise coefficients for dp/dt
plot_coefficients(
    coeffs,
    "eta_lambda",
    ["dpdt_coeff_x", "dpdt_coeff_p"],
)
