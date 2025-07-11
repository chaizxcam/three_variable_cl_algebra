from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coeffs = pd.read_csv("numerical_derivative_scan_params.csv")
coeffs = coeffs.map(lambda x: complex(x) if isinstance(x, str) and "j" in x else x)

coeffs["lambda_eff_factor"] = (
    coeffs["dpdt_coeff_p"].apply(np.real) * coeffs["eta_lambda"]
)


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


def plot_2d_real_imag_heatmaps(
    coeffs: pd.DataFrame,
    param_x: str,
    param_y: str,
    value_col: str,
    log_scale: bool = True,
) -> None:
    # Sort and prepare grid
    x_vals = np.sort(coeffs[param_x].unique())
    y_vals = np.sort(coeffs[param_y].unique())
    X, Y = np.meshgrid(x_vals, y_vals)

    # Real part
    real_pivot = coeffs.pivot(index=param_y, columns=param_x, values=value_col).apply(
        np.real
    )
    Z_real = real_pivot.values.astype(float)

    # Imaginary part
    imag_pivot = coeffs.pivot(index=param_y, columns=param_x, values=value_col).apply(
        np.imag
    )
    Z_imag = imag_pivot.values.astype(float)

    # --- Plot real part ---
    plt.figure(figsize=(8, 6))
    real_plot = plt.pcolormesh(X, Y, Z_real, shading="auto", cmap="viridis")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.colorbar(real_plot, label=f"Re({value_col})")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.title(f"Real part of {value_col}")
    plt.tight_layout()
    plt.savefig(f"heatmap_Re_{value_col}_vs_{param_x}_and_{param_y}.png", dpi=300)

    # --- Plot imaginary part ---
    plt.figure(figsize=(8, 6))
    imag_plot = plt.pcolormesh(X, Y, Z_imag, shading="auto", cmap="plasma")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.colorbar(imag_plot, label=f"Im({value_col})")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.title(f"Imaginary part of {value_col}")
    plt.tight_layout()
    plt.savefig(f"heatmap_Im_{value_col}_vs_{param_x}_and_{param_y}.png", dpi=300)
    plt.show()


plot_2d_real_imag_heatmaps(
    coeffs=coeffs, param_x="eta_omega", param_y="eta_m", value_col="lambda_eff_factor"
)
# # Plot noise coefficients for dx/dt
# plot_coefficients(
#     coeffs,
#     "eta_lambda",
#     ["dxdt_coeff_x", "dxdt_coeff_p"],
# )

# # Plot noise coefficients for dp/dt
# plot_coefficients(
#     coeffs,
#     "eta_lambda",
#     ["dpdt_coeff_x", "dpdt_coeff_p"],
# )
