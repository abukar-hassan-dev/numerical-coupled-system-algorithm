"""
SIMPLE Algorithm Solver for 2D Incompressible Navier-Stokes Equations.

Implements pressure-velocity coupling using the SIMPLE (Semi-Implicit Method
for Pressure-Linked Equations) algorithm on a staggered collocated grid,
with Rhie-Chow interpolation to suppress pressure-velocity decoupling.

Supported schemes:   FOU_CD (First-Order Upwind / Central Differencing)
                     Hybrid (switches automatically based on local Peclet number)
Supported solvers:   Gauss-Seidel, TDMA (Thomas Algorithm)
Pressure correction: noCorr, equiCorr, nonEquiCorr (Rhie-Chow)
"""

import sys
import time
import numpy as np


class SIMPLESolver:
    """
    Solves 2D steady incompressible laminar flow using the SIMPLE algorithm.

    Parameters
    ----------
    config : dict
        Solver configuration. See Config.defaults() for available keys.
    grid : Grid
        Parsed mesh and boundary condition data.
    """

    def __init__(self, config: dict, grid: "Grid"):
        self.cfg = config
        self.grid = grid
        self._allocate_arrays()
        self._init_boundary_conditions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the SIMPLE iteration loop.

        Returns
        -------
        dict
            Solution fields: u, v, p, and residual histories.
        """
        cfg = self.cfg
        start = time.time()

        for iteration in range(cfg["nSIMPLEiter"]):
            self._compute_momentum_coefficients()
            self._apply_underrelaxation()
            self._solve_momentum()
            self._update_outlet_fluxes()
            self._compute_face_fluxes()
            self._compute_pressure_correction_coefficients()
            self._solve_pressure_correction()
            self._correct_pressure()
            self._correct_velocities()
            self._correct_face_fluxes()
            self._compute_residuals()
            self._print_residuals(iteration)

            if self._converged():
                break

        elapsed = time.time() - start
        print(f"\nConverged in {iteration + 1} iterations | Wall time: {elapsed:.2f}s")

        return {
            "u": self.u, "v": self.v, "p": self.p,
            "res_u": self.res_u, "res_v": self.res_v, "res_c": self.res_c,
            "nodeX": self.grid.nodeX, "nodeY": self.grid.nodeY,
            "uTask2": self.grid.uTask2, "vTask2": self.grid.vTask2,
        }

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _allocate_arrays(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ
        nan = float("nan")

        def field():
            return np.zeros((nI, nJ)) * nan

        # Velocity and pressure
        self.u  = np.zeros((nI, nJ)) * nan
        self.v  = np.zeros((nI, nJ)) * nan
        self.p  = np.zeros((nI, nJ)) * nan
        self.pp = np.zeros((nI, nJ)) * nan

        # Momentum equation coefficients
        self.aE_uv = field(); self.aW_uv = field()
        self.aN_uv = field(); self.aS_uv = field()
        self.aP_uv = field()
        self.Su_u  = field(); self.Su_v  = field()

        # Pressure correction coefficients
        self.aE_pp = field(); self.aW_pp = field()
        self.aN_pp = field(); self.aS_pp = field()
        self.aP_pp = field(); self.Su_pp = field()

        # Face diffusion and convection coefficients
        self.De = field(); self.Dw = field()
        self.Dn = field(); self.Ds = field()
        self.Fe = np.zeros((nI, nJ)) * nan
        self.Fw = np.zeros((nI, nJ)) * nan
        self.Fn = np.zeros((nI, nJ)) * nan
        self.Fs = np.zeros((nI, nJ)) * nan

        # Pressure-correction d-coefficients (for velocity correction)
        self.de = field(); self.dw = field()
        self.dn = field(); self.ds = field()
        self.du = field(); self.dv = field()

        # Face pressure corrections
        self.pp_e = field(); self.pp_w = field()
        self.pp_n = field(); self.pp_s = field()

        # TDMA arrays
        self.P = field(); self.Q = field()

        # Rhie-Chow intermediates
        self.u_d_p = field(); self.v_d_p = field()
        self.u_d_e = field(); self.u_d_w = field()
        self.v_d_s = field(); self.v_d_n = field()
        self.p_g_e = field(); self.p_g_w = field()
        self.p_g_n = field(); self.p_g_s = field()

        # Outlet marker and residuals
        self.outlet = np.zeros((nI, nJ))
        self.res_u: list = []
        self.res_v: list = []
        self.res_c: list = []

    def _init_boundary_conditions(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ
        rho = self.cfg["rho"]

        self.u[:, :] = 0; self.v[:, :] = 0; self.p[:, :] = 0
        self.Fe[1:nI-1, 1:nJ-1] = 0
        self.Fw[1:nI-1, 1:nJ-1] = 0
        self.Fn[1:nI-1, 1:nJ-1] = 0
        self.Fs[1:nI-1, 1:nJ-1] = 0

        for i in range(1, nI-1):
            self.u[i, 0]    = g.uTask2[i, 0];    self.v[i, 0]    = g.vTask2[i, 0]
            self.u[i, nJ-1] = g.uTask2[i, nJ-1]; self.v[i, nJ-1] = g.vTask2[i, nJ-1]
            self.Fs[i, 1]      = self.v[i, 0]    * rho * g.dx_we[i, 1]
            self.Fn[i, nJ-2]   = self.v[i, nJ-1] * rho * g.dx_we[i, nJ-2]

        for j in range(1, nJ-1):
            self.u[0, j]    = g.uTask2[0, j];    self.v[0, j]    = g.vTask2[0, j]
            self.u[nI-1, j] = g.uTask2[nI-1, j]; self.v[nI-1, j] = g.vTask2[nI-1, j]
            self.Fw[1, j]      = self.u[0, j]    * rho * g.dy_sn[1, j]
            self.Fe[nI-2, j]   = self.u[nI-1, j] * rho * g.dy_sn[nI-2, j]

        for i in range(1, nI-1):
            if self.v[i, 0]    < 0: self.outlet[i, 0]    = 1
            if self.v[i, nJ-1] > 0: self.outlet[i, nJ-1] = 1
        for j in range(1, nJ-1):
            if self.u[0, j]    < 0: self.outlet[0, j]    = 1
            if self.u[nI-1, j] > 0: self.outlet[nI-1, j] = 1

        self._precompute_diffusion_coefficients()

    def _precompute_diffusion_coefficients(self):
        g = self.grid
        mu = self.cfg["mu"]
        for i in range(1, g.nI-1):
            for j in range(1, g.nJ-1):
                self.De[i,j] = mu * g.dy_sn[i,j] / g.dx_PE[i,j]
                self.Dw[i,j] = mu * g.dy_sn[i,j] / g.dx_WP[i,j]
                self.Dn[i,j] = mu * g.dx_we[i,j] / g.dy_PN[i,j]
                self.Ds[i,j] = mu * g.dx_we[i,j] / g.dy_SP[i,j]

    # ------------------------------------------------------------------
    # SIMPLE steps
    # ------------------------------------------------------------------

    def _compute_momentum_coefficients(self):
        g = self.grid
        scheme = self.cfg["scheme"]
        nI, nJ = g.nI, g.nJ

        if scheme == "FOU_CD":
            for i in range(1, nI-1):
                for j in range(1, nJ-1):
                    self.aE_uv[i,j] = (self.De[i,j] + max(0, -self.Fe[i,j])) * (1 - self.outlet[i+1,j])
                    self.aW_uv[i,j] = (self.Dw[i,j] + max(self.Fw[i,j], 0))  * (1 - self.outlet[i-1,j])
                    self.aN_uv[i,j] = (self.Dn[i,j] + max(0, -self.Fn[i,j])) * (1 - self.outlet[i,j+1])
                    self.aS_uv[i,j] = (self.Ds[i,j] + max(self.Fs[i,j], 0))  * (1 - self.outlet[i,j-1])
        elif scheme == "Hybrid":
            for i in range(1, nI-1):
                for j in range(1, nJ-1):
                    self.aE_uv[i,j] = max(-self.Fe[i,j], self.De[i,j] - g.fxe[i,j]*self.Fe[i,j], 0) * (1 - self.outlet[i+1,j])
                    self.aW_uv[i,j] = max( self.Fw[i,j], self.Dw[i,j] + g.fxw[i,j]*self.Fw[i,j], 0) * (1 - self.outlet[i-1,j])
                    self.aN_uv[i,j] = max(-self.Fn[i,j], self.Dn[i,j] - g.fyn[i,j]*self.Fn[i,j], 0) * (1 - self.outlet[i,j+1])
                    self.aS_uv[i,j] = max( self.Fs[i,j], self.Ds[i,j] + g.fys[i,j]*self.Fs[i,j], 0) * (1 - self.outlet[i,j-1])
        else:
            sys.exit(f"Unknown scheme: {scheme}")

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.aP_uv[i,j] = (self.aW_uv[i,j] + self.aE_uv[i,j] +
                                    self.aS_uv[i,j] + self.aN_uv[i,j])

    def _apply_underrelaxation(self):
        g = self.grid
        alpha = self.cfg["alphaUV"]
        nI, nJ = g.nI, g.nJ
        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.Su_u[i,j] = (((self.p[i-1,j] - self.p[i+1,j]) /
                                    (g.dx_WP[i,j] + g.dx_PE[i,j])) *
                                   g.dx_we[i,j] * g.dy_sn[i,j])
                self.Su_v[i,j] = (((self.p[i,j-1] - self.p[i,j+1]) /
                                    (g.dy_SP[i,j] + g.dy_PN[i,j])) *
                                   g.dx_we[i,j] * g.dy_sn[i,j])
                self.aP_uv[i,j] = self.aP_uv[i,j] / alpha
                self.Su_u[i,j] += (1 - alpha) * self.aP_uv[i,j] * self.u[i,j]
                self.Su_v[i,j] += (1 - alpha) * self.aP_uv[i,j] * self.v[i,j]

    def _solve_momentum(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ
        for _ in range(self.cfg["nLinSolIter_uv"]):
            for i in range(1, nI-1):
                for j in range(1, nJ-1):
                    self.u[i,j] = ((self.aW_uv[i,j]*self.u[i-1,j] + self.aE_uv[i,j]*self.u[i+1,j] +
                                    self.aS_uv[i,j]*self.u[i,j-1] + self.aN_uv[i,j]*self.u[i,j+1] +
                                    self.Su_u[i,j]) / self.aP_uv[i,j])
            for i in range(1, nI-1):
                for j in range(1, nJ-1):
                    self.v[i,j] = ((self.aW_uv[i,j]*self.v[i-1,j] + self.aE_uv[i,j]*self.v[i+1,j] +
                                    self.aS_uv[i,j]*self.v[i,j-1] + self.aN_uv[i,j]*self.v[i,j+1] +
                                    self.Su_v[i,j]) / self.aP_uv[i,j])

    def _update_outlet_fluxes(self):
        g = self.grid
        rho = self.cfg["rho"]
        nI, nJ = g.nI, g.nJ

        for i in range(1, nI-1):
            if self.outlet[i, 0]:
                self.u[i,0] = self.u[i,1]; self.v[i,0] = self.v[i,1]
                self.Fs[i,1] = rho * self.v[i,0] * g.dx_we[i,1]
            if self.outlet[i, nJ-1]:
                self.u[i,nJ-1] = self.u[i,nJ-2]; self.v[i,nJ-1] = self.v[i,nJ-2]
                self.Fn[i,nJ-2] = rho * self.v[i,nJ-1] * g.dx_we[i,nJ-2]
        for j in range(1, nJ-1):
            if self.outlet[0, j]:
                self.u[0,j] = self.u[1,j]; self.v[0,j] = self.v[1,j]
                self.Fw[1,j] = rho * self.u[0,j] * g.dy_sn[1,j]
            if self.outlet[nI-1, j]:
                self.u[nI-1,j] = self.u[nI-2,j]; self.v[nI-1,j] = self.v[nI-2,j]
                self.Fe[nI-2,j] = rho * self.u[nI-1,j] * g.dy_sn[nI-2,j]

        # Scale outlet fluxes to enforce global continuity
        inletFlux = (- np.nansum((1-self.outlet[nI-1,1:nJ-1]) * self.Fe[nI-2,1:nJ-1])
                     + np.nansum((1-self.outlet[0,1:nJ-1])    * self.Fw[1,1:nJ-1])
                     - np.nansum((1-self.outlet[1:nI-1,nJ-1]) * self.Fn[1:nI-1,nJ-2])
                     + np.nansum((1-self.outlet[1:nI-1,0])    * self.Fs[1:nI-1,1]))

        outletFlux = (  np.nansum(self.outlet[nI-1,1:nJ-1] * self.Fe[nI-2,1:nJ-1])
                      - np.nansum(self.outlet[0,1:nJ-1]    * self.Fw[1,1:nJ-1])
                      + np.nansum(self.outlet[1:nI-1,nJ-1] * self.Fn[1:nI-1,nJ-2])
                      - np.nansum(self.outlet[1:nI-1,0]    * self.Fs[1:nI-1,1]))

        if outletFlux == 0:
            # First iteration: outlet velocities are zero, initialise from area
            outletArea = (np.nansum(self.outlet[nI-1,1:nJ-1] * g.dy_sn[nI-2,1:nJ-1])
                        + np.nansum(self.outlet[0,1:nJ-1]    * g.dy_sn[1,1:nJ-1])
                        + np.nansum(self.outlet[1:nI-1,nJ-1] * g.dx_we[1:nI-1,nJ-2])
                        + np.nansum(self.outlet[1:nI-1,0]    * g.dx_we[1:nI-1,1]))
            for i in range(1, nI-1):
                if self.outlet[i,0]:
                    self.Fs[i,1] = -inletFlux * g.dx_we[i,1] / outletArea
                if self.outlet[i,nJ-1]:
                    self.Fn[i,nJ-2] = inletFlux * g.dx_we[i,nJ-2] / outletArea
            for j in range(1, nJ-1):
                if self.outlet[0,j]:
                    self.Fw[1,j] = -inletFlux * g.dy_sn[1,j] / outletArea
                if self.outlet[nI-1,j]:
                    self.Fe[nI-2,j] = inletFlux * g.dy_sn[nI-2,j] / outletArea
        else:
            scale = inletFlux / outletFlux
            for i in range(1, nI-1):
                if self.outlet[i,0]:
                    self.v[i,0] *= scale; self.Fs[i,1] *= scale
                if self.outlet[i,nJ-1]:
                    self.v[i,nJ-1] *= scale; self.Fn[i,nJ-2] *= scale
            for j in range(1, nJ-1):
                if self.outlet[0,j]:
                    self.u[0,j] *= scale; self.Fw[1,j] *= scale
                if self.outlet[nI-1,j]:
                    self.u[nI-1,j] *= scale; self.Fe[nI-2,j] *= scale

    def _compute_face_fluxes(self):
        g = self.grid
        rho = self.cfg["rho"]
        mode = self.cfg["RhieChow"]
        nI, nJ = g.nI, g.nJ

        if mode == "noCorr":
            for i in range(1, nI-2):
                for j in range(1, nJ-1):
                    self.Fe[i,j] = rho * g.dy_sn[i,j] * (g.fxe[i,j]*self.u[i+1,j] + (1-g.fxe[i,j])*self.u[i,j])
            for i in range(2, nI-1):
                for j in range(1, nJ-1):
                    self.Fw[i,j] = rho * g.dy_sn[i,j] * (g.fxw[i,j]*self.u[i-1,j] + (1-g.fxw[i,j])*self.u[i,j])
            for i in range(1, nI-1):
                for j in range(1, nJ-2):
                    self.Fn[i,j] = rho * g.dx_we[i,j] * (g.fyn[i,j]*self.v[i,j+1] + (1-g.fyn[i,j])*self.v[i,j])
            for i in range(1, nI-1):
                for j in range(2, nJ-1):
                    self.Fs[i,j] = rho * g.dx_we[i,j] * (g.fys[i,j]*self.v[i,j-1] + (1-g.fys[i,j])*self.v[i,j])

        elif mode == "equiCorr":
            for i in range(1, nI-2):
                for j in range(1, nJ-1):
                    u_bar = g.fxe[i,j]*self.u[i+1,j] + (1-g.fxe[i,j])*self.u[i,j]
                    d_bar = g.dy_sn[i,j] / (g.fxe[i,j]*self.aP_uv[i+1,j] + (1-g.fxe[i,j])*self.aP_uv[i,j])
                    dp    = (self.p[i,j] - self.p[i+1,j]) / g.dx_PE[i,j]
                    dp_n  = ((self.p[i-1,j]-self.p[i+1,j])/(g.dx_WP[i,j]+g.dx_PE[i,j])*g.fxe[i,j] +
                             (self.p[i,j]-self.p[i+2,j])/(g.dx_WP[i+1,j]+g.dx_PE[i+1,j])*(1-g.fxe[i,j]))
                    self.Fe[i,j] = rho * g.dy_sn[i,j] * (u_bar + d_bar*(dp - dp_n))
            for i in range(2, nI-1):
                for j in range(1, nJ-1):
                    self.Fw[i,j] = self.Fe[i-1,j]
            for i in range(1, nI-1):
                for j in range(1, nJ-2):
                    v_bar = g.fyn[i,j]*self.v[i,j+1] + (1-g.fyn[i,j])*self.v[i,j]
                    d_bar = g.dx_we[i,j] / (g.fyn[i,j]*self.aP_uv[i,j+1] + (1-g.fyn[i,j])*self.aP_uv[i,j])
                    dp    = (self.p[i,j] - self.p[i,j+1]) / g.dy_PN[i,j]
                    dp_n  = ((self.p[i,j-1]-self.p[i,j+1])/(g.dy_SP[i,j]+g.dy_PN[i,j])*g.fyn[i,j] +
                             (self.p[i,j]-self.p[i,j+2])/(g.dy_SP[i,j+1]+g.dy_PN[i,j+1])*(1-g.fyn[i,j]))
                    self.Fn[i,j] = rho * g.dx_we[i,j] * (v_bar + d_bar*(dp - dp_n))
            for i in range(1, nI-1):
                for j in range(2, nJ-1):
                    self.Fs[i,j] = self.Fn[i,j-1]

        elif mode == "nonEquiCorr":
            for i in range(1, nI-2):
                for j in range(1, nJ-1):
                    self.u_d_p[i,j] = (self.u[i,j] - (g.dx_we[i,j]*g.dy_sn[i,j]/self.aP_uv[i,j]) *
                                        ((self.p[i-1,j]-self.p[i+1,j])/(g.dx_WP[i,j]+g.dx_PE[i,j])))
                    self.u_d_e[i,j] = (g.fxe[i,j] * (self.u[i+1,j] -
                                        (g.dx_we[i+1,j]*g.dy_sn[i,j]/self.aP_uv[i+1,j]) *
                                        ((self.p[i,j]-self.p[i+2,j])/(g.dx_PE[i+1,j]+g.dx_PE[i,j]))) +
                                       (1-g.fxe[i,j]) * self.u_d_p[i,j])
                    self.p_g_e[i,j] = (self.p[i,j] - self.p[i+1,j]) / g.dx_PE[i,j]
                    self.Fe[i,j] = rho * g.dy_sn[i,j] * (self.u_d_e[i,j] +
                                    ((g.dx_we[i+1,j]*g.dy_sn[i+1,j]/self.aP_uv[i+1,j])*g.fxe[i,j] +
                                     (1-g.fxe[i,j])*(g.dx_we[i,j]*g.dy_sn[i,j]/self.aP_uv[i,j])) *
                                    self.p_g_e[i,j])
            for i in range(1, nI-1):
                for j in range(1, nJ-2):
                    self.v_d_p[i,j] = (self.v[i,j] - (g.dx_we[i,j]*g.dy_sn[i,j]/self.aP_uv[i,j]) *
                                        ((self.p[i,j-1]-self.p[i,j+1])/(g.dy_SP[i,j]+g.dy_PN[i,j])))
                    self.v_d_n[i,j] = (g.fyn[i,j] * ((self.v[i,j+1] -
                                        (g.dx_we[i,j]*g.dy_sn[i,j+1]/self.aP_uv[i,j+1]) *
                                        ((self.p[i,j]-self.p[i,j+2])/(g.dy_PN[i,j+1]+g.dy_PN[i,j]))) +
                                       (1-g.fyn[i,j])*self.v_d_p[i,j]) + (1-g.fyn[i,j])*self.v_d_p[i,j])
                    self.p_g_n[i,j] = (self.p[i,j] - self.p[i,j+1]) / g.dy_PN[i,j]
                    self.Fn[i,j] = rho * g.dx_we[i,j] * (self.v_d_n[i,j] +
                                    ((g.dx_we[i,j+1]*g.dy_sn[i,j+1]/self.aP_uv[i,j+1])*g.fyn[i,j] +
                                     (1-g.fyn[i,j])*(g.dx_we[i,j]*g.dy_sn[i,j]/self.aP_uv[i,j])) *
                                    self.p_g_n[i,j])
            for i in range(2, nI-1):
                for j in range(1, nJ-1):
                    self.Fw[i,j] = self.Fe[i-1,j]
            for i in range(1, nI-1):
                for j in range(2, nJ-1):
                    self.Fs[i,j] = self.Fn[i,j-1]
        else:
            sys.exit(f"Unknown RhieChow option: {mode}")

    def _compute_pressure_correction_coefficients(self):
        g = self.grid
        rho = self.cfg["rho"]
        nI, nJ = g.nI, g.nJ

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.dw[i,j] = g.dy_sn[i,j] / (self.aP_uv[i-1,j]*g.fxw[i,j] + (1-g.fxw[i,j])*self.aP_uv[i,j])
                self.de[i,j] = g.dy_sn[i,j] / (self.aP_uv[i+1,j]*g.fxe[i,j] + (1-g.fxe[i,j])*self.aP_uv[i,j])
                self.ds[i,j] = g.dx_we[i,j] / (self.aP_uv[i,j-1]*g.fys[i,j] + (1-g.fys[i,j])*self.aP_uv[i,j])
                self.dn[i,j] = g.dx_we[i,j] / (self.aP_uv[i,j+1]*g.fyn[i,j] + (1-g.fyn[i,j])*self.aP_uv[i,j])
                self.du[i,j] = g.dy_sn[i,j] / self.aP_uv[i,j]
                self.dv[i,j] = g.dx_we[i,j] / self.aP_uv[i,j]
                self.aW_pp[i,j] = rho * self.dw[i,j] * g.dy_sn[i,j]
                self.aE_pp[i,j] = rho * self.de[i,j] * g.dy_sn[i,j]
                self.aS_pp[i,j] = rho * self.ds[i,j] * g.dx_we[i,j]
                self.aN_pp[i,j] = rho * self.dn[i,j] * g.dx_we[i,j]

        # Homogeneous Neumann at boundaries (no flux through walls)
        for j in range(1, nJ-1):
            self.aE_pp[nI-2,j] = 0; self.de[nI-2,j] = 0
            self.aW_pp[1,j]    = 0; self.dw[1,j]    = 0
        for i in range(1, nI-1):
            self.aN_pp[i,nJ-2] = 0; self.dn[i,nJ-2] = 0
            self.aS_pp[i,1]    = 0; self.ds[i,1]    = 0

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.aP_pp[i,j] = (self.aW_pp[i,j] + self.aE_pp[i,j] +
                                    self.aS_pp[i,j] + self.aN_pp[i,j])
                self.Su_pp[i,j] = ((self.Fw[i,j] - self.Fe[i,j]) +
                                    (self.Fs[i,j] - self.Fn[i,j]))

    def _solve_pressure_correction(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ
        solver = self.cfg["linSol_pp"]
        self.pp[:, :] = 0

        if solver == "Gauss-Seidel":
            for _ in range(self.cfg["nLinSolIter_pp"]):
                for i in range(1, nI-1):
                    for j in range(1, nJ-1):
                        self.pp[i,j] = ((self.aW_pp[i,j]*self.pp[i-1,j] + self.aE_pp[i,j]*self.pp[i+1,j] +
                                         self.aS_pp[i,j]*self.pp[i,j-1] + self.aN_pp[i,j]*self.pp[i,j+1] +
                                         self.Su_pp[i,j]) / self.aP_pp[i,j])

        elif solver == "TDMA":
            for _ in range(self.cfg["nLinSolIter_pp"]):
                for j in range(1, nJ-1):
                    for i in range(1, nI-1):
                        rhs = (self.aN_pp[i,j]*self.pp[i,j+1] + self.aS_pp[i,j]*self.pp[i,j-1] +
                               self.Su_pp[i,j] + self.aW_pp[i,j]*self.pp[i-1,j])
                        if i == 1:
                            self.P[i,j] = self.aE_pp[i,j] / self.aP_pp[i,j]
                            self.Q[i,j] = rhs / self.aP_pp[i,j]
                        elif i < nI-1:
                            denom = self.aP_pp[i,j] - self.aW_pp[i,j]*self.P[i-1,j]
                            self.P[i,j] = self.aE_pp[i,j] / denom
                            self.Q[i,j] = (rhs + self.aW_pp[i,j]*self.Q[i-1,j]) / denom
                        if i == nI-2:
                            self.P[i,j] = 0
                            extra = self.aE_pp[i,j]*self.pp[i+1,j]
                            self.Q[i,j] = (rhs + self.aW_pp[i,j]*self.Q[i-1,j] + extra) / (self.aP_pp[i,j] - self.aW_pp[i,j]*self.P[i-1,j])
                    for i in range(nI-2, 0, -1):
                        self.pp[i,j] = self.P[i,j]*self.pp[i+1,j] + self.Q[i,j]
        else:
            sys.exit(f"Unknown linear solver: {solver}")

        # Fix pressure level and apply Neumann BCs
        self.pp[:, :] -= self.pp[self.cfg["pRef_i"], self.cfg["pRef_j"]]
        self.pp[:,  0] = self.pp[:,  1]; self.pp[:, -1] = self.pp[:, -2]
        self.pp[0,  :] = self.pp[1,  :]; self.pp[-1, :] = self.pp[-2, :]

    def _correct_pressure(self):
        g = self.grid
        alpha = self.cfg["alphaP"]
        nI, nJ = g.nI, g.nJ

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.p[i,j] += alpha * self.pp[i,j]

        # Extrapolate pressure to boundaries with constant gradient
        for j in range(1, nJ-1):
            i = nI-1
            self.p[i,j] = self.p[i-1,j] + ((self.p[i-1,j]-self.p[i-2,j])/g.dx_PE[i-2,j])*g.dx_PE[i-1,j]
            i = 0
            self.p[i,j] = self.p[i+1,j] - ((self.p[i+2,j]-self.p[i+1,j])/g.dx_WP[i+2,j])*g.dx_WP[i+1,j]
        for i in range(1, nI-1):
            j = nJ-1
            self.p[i,j] = self.p[i,j-1] + ((self.p[i,j-1]-self.p[i,j-2])/g.dy_PN[i,j-2])*g.dy_PN[i,j-1]
            j = 0
            self.p[i,j] = self.p[i,j+1] - ((self.p[i,j+2]-self.p[i,j+1])/g.dy_SP[i,j+2])*g.dy_SP[i,j+1]

        # Interpolate pressure to corners
        self.p[0,0]       = (self.p[1,0]     + self.p[0,1])     / 2
        self.p[nI-1,0]    = (self.p[nI-2,0]  + self.p[nI-1,1])  / 2
        self.p[nI-1,nJ-1] = (self.p[nI-1,nJ-2] + self.p[nI-2,nJ-1]) / 2
        self.p[0,nJ-1]    = (self.p[1,nJ-1]  + self.p[0,nJ-2])  / 2

    def _correct_velocities(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                if i == nI-2:
                    self.pp_e[i,j] = self.pp[i+1,j]
                else:
                    self.pp_e[i,j] = (1-g.fxe[i,j])*self.pp[i,j] + g.fxe[i,j]*self.pp[i+1,j]
                if i == 1:
                    self.pp_w[i,j] = self.pp[i-1,j]
                else:
                    self.pp_w[i,j] = (1-g.fxw[i,j])*self.pp[i,j] + g.fxw[i,j]*self.pp[i-1,j]
                if j == nJ-2:
                    self.pp_n[i,j] = self.pp[i,j+1]
                else:
                    self.pp_n[i,j] = (1-g.fyn[i,j])*self.pp[i,j] + g.fyn[i,j]*self.pp[i,j+1]
                if j == 1:
                    self.pp_s[i,j] = self.pp[i,j-1]
                else:
                    self.pp_s[i,j] = (1-g.fys[i,j])*self.pp[i,j] + g.fys[i,j]*self.pp[i,j-1]

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                self.u[i,j] += self.du[i,j] * (self.pp_w[i,j] - self.pp_e[i,j])
                self.v[i,j] += self.dv[i,j] * (self.pp_s[i,j] - self.pp_n[i,j])

    def _correct_face_fluxes(self):
        g = self.grid
        rho = self.cfg["rho"]
        nI, nJ = g.nI, g.nJ

        for i in range(1, nI-2):
            for j in range(1, nJ-1):
                self.Fe[i,j] += rho * self.de[i,j] * (self.pp[i,j]-self.pp[i+1,j]) * g.dy_sn[i,j]
        for i in range(2, nI-1):
            for j in range(1, nJ-1):
                self.Fw[i,j] += rho * self.dw[i,j] * (self.pp[i-1,j]-self.pp[i,j]) * g.dy_sn[i,j]
        for i in range(1, nI-1):
            for j in range(1, nJ-2):
                self.Fn[i,j] += rho * self.dn[i,j] * (self.pp[i,j]-self.pp[i,j+1]) * g.dx_we[i,j]
        for i in range(1, nI-1):
            for j in range(2, nJ-1):
                self.Fs[i,j] += rho * self.ds[i,j] * (self.pp[i,j-1]-self.pp[i,j]) * g.dx_we[i,j]

    def _compute_residuals(self):
        g = self.grid
        nI, nJ = g.nI, g.nJ
        ru = rv = rc = 0.0

        for i in range(1, nI-1):
            for j in range(1, nJ-1):
                ru += abs(self.aP_uv[i,j]*self.u[i,j] -
                          (self.aE_uv[i,j]*self.u[i+1,j] + self.aW_uv[i,j]*self.u[i-1,j] +
                           self.aN_uv[i,j]*self.u[i,j+1] + self.aS_uv[i,j]*self.u[i,j-1] +
                           self.Su_u[i,j]))
                rv += abs(self.aP_uv[i,j]*self.v[i,j] -
                          (self.aE_uv[i,j]*self.v[i+1,j] + self.aW_uv[i,j]*self.v[i-1,j] +
                           self.aN_uv[i,j]*self.v[i,j+1] + self.aS_uv[i,j]*self.v[i,j-1] +
                           self.Su_v[i,j]))
                rc += abs((self.Fw[i,j]-self.Fe[i,j]) + (self.Fs[i,j]-self.Fn[i,j]))

        self.res_u.append(ru); self.res_v.append(rv); self.res_c.append(rc)

        # Normalise against first iteration
        self.res_u[-1] /= self.res_u[0]
        self.res_v[-1] /= self.res_v[0]
        self.res_c[-1] /= self.res_c[0]

    def _print_residuals(self, iteration: int):
        print(f"Iter: {iteration:5d} | resU = {self.res_u[-1]:.4e} | "
              f"resV = {self.res_v[-1]:.4e} | resCon = {self.res_c[-1]:.4e}")

    def _converged(self) -> bool:
        tol = self.cfg["resTol"]
        return max(self.res_u[-1], self.res_v[-1], self.res_c[-1]) < tol
