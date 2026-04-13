"""
scripts/run_solver.py
=====================
Entry point for the 2D Navier-Stokes SIMPLE solver.

Usage
-----
    python scripts/run_solver.py

Edit the CONFIG dictionary below to change case, grid type, fluid properties,
or solver settings. All tuneable parameters live here — no need to touch src/.
"""

import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid import Grid
from src.solver import SIMPLESolver
from src.postprocess import save_all_plots


# ============================================================
# Configuration — edit here
# ============================================================
CONFIG = {
    # Mesh
    "grid_type": "coarse",   # 'coarse' or 'fine'
    "case_id":    13,         # 1–25

    # Fluid properties
    "rho":  1.0,
    "mu":   0.001,

    # SIMPLE iteration control
    "nSIMPLEiter":    1000,
    "nLinSolIter_pp": 10,
    "nLinSolIter_uv": 3,
    "resTol":         1e-3,

    # Under-relaxation factors
    "alphaUV": 0.7,
    "alphaP":  0.3,

    # Solver and scheme selection
    "linSol_pp": "TDMA",        # 'TDMA' or 'Gauss-Seidel'
    "scheme":    "Hybrid",      # 'Hybrid' or 'FOU_CD'
    "RhieChow":  "nonEquiCorr", # 'nonEquiCorr', 'equiCorr', or 'noCorr'

    # Reference pressure node (interior node only)
    "pRef_i": 3,
    "pRef_j": 3,
}
# ============================================================


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    print(f"Loading grid  : case {CONFIG['case_id']} ({CONFIG['grid_type']})")
    grid = Grid(data_dir, CONFIG["case_id"], CONFIG["grid_type"])

    print(f"Scheme        : {CONFIG['scheme']}")
    print(f"Linear solver : {CONFIG['linSol_pp']}")
    print(f"Rhie-Chow     : {CONFIG['RhieChow']}")
    print("-" * 60)

    solver   = SIMPLESolver(CONFIG, grid)
    solution = solver.run()

    save_all_plots(
        solution,
        case_id   = CONFIG["case_id"],
        grid_type = CONFIG["grid_type"],
        pRef_i    = CONFIG["pRef_i"],
        pRef_j    = CONFIG["pRef_j"],
    )


if __name__ == "__main__":
    main()
