"""
Grid loader for the 2D Navier-Stokes SIMPLE solver.

Reads mesh coordinates and Task-2 reference velocity fields from the
standard data directory layout used in Chalmers MTF073.
"""

import numpy as np
from pathlib import Path


class Grid:
    """
    Parses mesh data and computes all geometric quantities needed by the solver.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing grid sub-folders (e.g. ``data/``).
    case_id : int
        Case number in range 1–25.
    grid_type : str
        Either ``'coarse'`` or ``'fine'``.
    """

    # Maps case_id → grid number (Chalmers MTF073 layout)
    _GRID_MAP = [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5]

    def __init__(self, data_dir: str | Path, case_id: int, grid_type: str = "coarse"):
        self.data_dir  = Path(data_dir)
        self.case_id   = case_id
        self.grid_type = grid_type
        self._load()

    # ------------------------------------------------------------------
    # Public attributes (set by _load)
    # ------------------------------------------------------------------
    #   nI, nJ          : number of nodes (incl. boundaries)
    #   nodeX, nodeY    : node coordinates
    #   pointX, pointY  : mesh-point coordinates
    #   dx_PE, dx_WP    : node-to-node distances in x
    #   dy_PN, dy_SP    : node-to-node distances in y
    #   dx_we, dy_sn    : control-volume face lengths
    #   fxe/fxw/fyn/fys : interpolation factors
    #   uTask2, vTask2  : reference velocity field (boundary conditions)
    # ------------------------------------------------------------------

    def _load(self):
        grid_num = self._GRID_MAP[self.case_id - 1]
        path = self.data_dir / f"grid{grid_num}" / f"{self.grid_type}_grid"

        xvec = np.genfromtxt(path / "xc.dat")
        yvec = np.genfromtxt(path / "yc.dat")
        u_d  = np.genfromtxt(path / "u.dat")
        v_d  = np.genfromtxt(path / "v.dat")

        mI, mJ   = len(xvec), len(yvec)
        nI, nJ   = mI + 1, mJ + 1
        self.nI, self.nJ = nI, nJ
        nan = float("nan")

        # Point coordinates
        self.pointX = np.zeros((mI, mJ))
        self.pointY = np.zeros((mI, mJ))
        for i in range(mI):
            for j in range(mJ):
                self.pointX[i,j] = xvec[i]
                self.pointY[i,j] = yvec[j]

        # Node coordinates
        self.nodeX = np.zeros((nI, nJ)) * nan
        self.nodeY = np.zeros((nI, nJ)) * nan
        for i in range(nI):
            for j in range(nJ):
                if 0 < i < nI-1:
                    self.nodeX[i,j] = 0.5*(xvec[i] + xvec[i-1])
                if 0 < j < nJ-1:
                    self.nodeY[i,j] = 0.5*(yvec[j] + yvec[j-1])
        self.nodeX[0,:]  = xvec[0];  self.nodeX[-1,:] = xvec[-1]
        self.nodeY[:,0]  = yvec[0];  self.nodeY[:,-1] = yvec[-1]

        # Inter-node distances and CV sizes
        def _field(): return np.zeros((nI, nJ)) * nan

        self.dx_PE = _field(); self.dx_WP = _field()
        self.dy_PN = _field(); self.dy_SP = _field()
        self.dx_we = _field(); self.dy_sn = _field()

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.dx_PE[i,j] = self.nodeX[i+1,j] - self.nodeX[i,j]
                self.dx_WP[i,j] = self.nodeX[i,j]   - self.nodeX[i-1,j]
                self.dy_PN[i,j] = self.nodeY[i,j+1] - self.nodeY[i,j]
                self.dy_SP[i,j] = self.nodeY[i,j]   - self.nodeY[i,j-1]
                self.dx_we[i,j] = self.pointX[i,j]  - self.pointX[i-1,j]
                self.dy_sn[i,j] = self.pointY[i,j]  - self.pointY[i,j-1]

        # Interpolation factors
        self.fxe = _field(); self.fxw = _field()
        self.fyn = _field(); self.fys = _field()

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.fxe[i,j] = 0.5 * self.dx_we[i,j] / self.dx_PE[i,j]
                self.fxw[i,j] = 0.5 * self.dx_we[i,j] / self.dx_WP[i,j]
                self.fyn[i,j] = 0.5 * self.dy_sn[i,j] / self.dy_PN[i,j]
                self.fys[i,j] = 0.5 * self.dy_sn[i,j] / self.dy_SP[i,j]

        # Reference velocity field (boundary conditions from Task 2)
        self.uTask2 = u_d.reshape(nI, nJ)
        self.vTask2 = v_d.reshape(nI, nJ)
        self.uTask2[self.uTask2 == 1e-10] = 0
        self.vTask2[self.vTask2 == 1e-10] = 0

        self.L = xvec[-1] - xvec[0]
        self.H = yvec[-1] - yvec[0]
