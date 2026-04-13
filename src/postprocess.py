"""
Post-processing and plotting utilities for the SIMPLE solver.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})


def save_all_plots(solution: dict, case_id: int, grid_type: str,
                   pRef_i: int, pRef_j: int, out_dir: str = "figures") -> None:
    """Generate and save all standard diagnostic plots.

    Parameters
    ----------
    solution : dict
        Output from ``SIMPLESolver.run()``.
    case_id : int
        Case number (used in file names).
    grid_type : str
        ``'coarse'`` or ``'fine'`` (used in file names).
    pRef_i, pRef_j : int
        Reference pressure node indices.
    out_dir : str
        Directory where figures are saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = f"Case_{case_id}_{grid_type}"

    u      = solution["u"]
    v      = solution["v"]
    p      = solution["p"]
    nodeX  = solution["nodeX"]
    nodeY  = solution["nodeY"]
    uTask2 = solution["uTask2"]
    vTask2 = solution["vTask2"]
    res_u  = solution["res_u"]
    res_v  = solution["res_v"]
    res_c  = solution["res_c"]

    nIter = len(res_u)

    # ---- Velocity vectors ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.quiver(nodeX.T, nodeY.T, u.T, v.T)
    ax.set_title("Velocity vectors")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_velocityVectors.png", dpi=150)
    plt.close(fig)

    # ---- U-velocity contour ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(nodeX.T, nodeY.T, u.T, cmap="coolwarm", levels=30)
    plt.colorbar(cf, ax=ax, label="[m/s]")
    ax.set_title("U velocity [m/s]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_uVelocityContour.png", dpi=150)
    plt.close(fig)

    # ---- V-velocity contour ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(nodeX.T, nodeY.T, v.T, cmap="coolwarm", levels=30)
    plt.colorbar(cf, ax=ax, label="[m/s]")
    ax.set_title("V velocity [m/s]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_vVelocityContour.png", dpi=150)
    plt.close(fig)

    # ---- Pressure contour ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(nodeX.T, nodeY.T, p.T, cmap="coolwarm", levels=30)
    plt.colorbar(cf, ax=ax, label="[Pa]")
    ax.plot(nodeX[pRef_i, pRef_j], nodeY[pRef_i, pRef_j], "bo",
            label="p reference node")
    ax.set_title("Pressure [Pa]")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_pressureContour.png", dpi=150)
    plt.close(fig)

    # ---- Velocity validation (horizontal centreline) ---------------------
    nJ  = nodeX.shape[1]
    mid = int(nJ / 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nodeX[:, mid], u[:, mid],      "b",    label="U")
    ax.plot(nodeX[:, mid], uTask2[:, mid], "b--",  label="U Task 2")
    ax.plot(nodeX[:, mid], v[:, mid],      "r",    label="V")
    ax.plot(nodeX[:, mid], vTask2[:, mid], "r--",  label="V Task 2")
    ax.set_title("Velocity validation — horizontal centreline")
    ax.set_xlabel("x [m]"); ax.set_ylabel("Velocity [m/s]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_velocityValidation.png", dpi=150)
    plt.close(fig)

    # ---- Residual convergence --------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(nIter), res_u, "blue",  label="U momentum")
    ax.semilogy(range(nIter), res_v, "red",   label="V momentum")
    ax.semilogy(range(nIter), res_c, "green", label="Continuity")
    ax.set_title("Residual convergence")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Normalised residual [−]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{tag}_residuals.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to '{out_dir}/'")
