"""
Implementation of the pressure-based coupling method.
"""

__author__ = "Koen Devesse"
__date__ = "January 4, 2024"

import numpy as np
from numba import njit
import numba
from scipy.interpolate import RegularGridInterpolator

from wayve.forcing.pressure_forcing.pressure_forcing import pressure_gradients
from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import VaryingBackground, \
    height_average_shape_function


class PressureBased(VaryingBackground):
    """
    A coupling based on the pressure-driven velocities. See Stipa et al., 2023 [1].

    References
    ----------
    .. [1]  Stipa, S., Ajay, A., Allaerts, D., and Brinkerhoff, J.: "The Multi-Scale Coupled Model: a New Framework
        Capturing Wind Farm-Atmosphere Interaction and Global Blockage Effects", Wind Energ. Sci. Discuss. [preprint],
        https://doi.org/10.5194/wes-2023-75, in review, 2023
    """

    def update_ub_vb(self, model, wind_farm, result):
        """
        Update the evaluator functions for u_b and v_b, based on the given APM state.
        """
        # APM components
        abl = model.abl
        grid = model.grid
        # Set up RHS vector
        rhs = set_up_rhs(model, result)
        # Apply APM Mx operation
        x_b = Mx(model, rhs)
        # Extract lower layer velocity components in Fourier space
        N = grid.N2
        u1b_c = x_b[:N]
        v1b_c = x_b[N:2*N]
        # Convert to real space
        u1b_r = grid.c2r(u1b_c.reshape(grid.shape2))
        v1b_r = grid.c2r(v1b_c.reshape(grid.shape2))
        # Select region around wind farm
        x_sel, y_sel, selection = self.select_grid_around_farm(wind_farm, grid)
        # Select variables
        shape_b = (len(x_sel), len(y_sel))
        u1b = np.reshape(u1b_r[selection], shape_b)
        v1b = np.reshape(v1b_r[selection], shape_b)
        eta_apm = np.reshape(result["eta1r"][selection], shape_b)
        # Vertical grid and shape function
        z0 = abl.zs[0]
        h1_max = abl.H1 + np.max(eta_apm)
        Nz = 100
        z = np.linspace(z0, h1_max, Nz, endpoint=True)
        f = self.vertical_profile(abl, z)
        # Height-averaged shape function
        h1 = abl.H1 + eta_apm
        f_ha = height_average_shape_function(f, z, h1)
        # Get background velocity scales u_b and v_b
        u_b = np.divide(u1b, f_ha)
        v_b = np.divide(v1b, f_ha)
        # Set up interpolation functions
        f_ub = RegularGridInterpolator((x_sel, y_sel), u_b, method="linear", bounds_error=False, fill_value=None)
        f_vb = RegularGridInterpolator((x_sel, y_sel), v_b, method="linear", bounds_error=False, fill_value=None)
        # Set up evaluator functions
        self.ub_evaluator = lambda x, y: f_ub((x, y))
        self.vb_evaluator = lambda x, y: f_vb((x, y))


def Mx(model, rhs):
    """
    Applies the M(x) operation to the pressure-based forcing vector.

    Parameters
    ----------
    model:  Model object
                APM object, providing access to ABL, grid, and parametrization information
    rhs: 1d numpy array
        input vector

    Returns
    -------
    _: 1d numpy array
        matrix vector product M*rhs
    """
    # APM components
    grid = model.grid   # Grid object
    abl = model.abl     # ABL object
    mfp = model.mfp     # Momentum flux parametrization object
    pressure = model.pressure
    # Numerical grid
    N = grid.N2
    ks = grid.ks2
    ls = grid.ls
    # Atmospheric conditions
    U1 = abl.U1
    V1 = abl.V1
    U2 = abl.U2
    V2 = abl.V2
    nu1 = abl.nu1
    nu2 = abl.nu2
    fc = abl.fc
    # Momentum flux Jacobian
    tau_jac = mfp.mf_term_Jac(abl)
    tau_jac_bg = tau_jac[:, :4]     # The eta-related terms are not included in the matrix system
    # Matrix pressure_par
    M = solve_coupling_system(ks, ls, U1, V1, U2, V2, nu1, nu2, fc, tau_jac_bg, N, rhs)
    # Defunct modes
    for var in range(4):
        grid.defunct_modes(M[var*N:(var+1)*N])
    return M


def set_up_rhs(model, result):
    """Set up the right-hand side for the pressure components of the APM equations"""
    # APM components
    abl = model.abl
    grid = model.grid
    mfp = model.mfp
    # Pressure contribution #
    # Pressure perturbations in Fourier space
    p1c = result["p1c"]
    p2c = result["p2c"]
    # Calculate pressure gradients
    Fu1, Fv1, Fu2, Fv2 = pressure_gradients(grid.ks2, grid.ls, p1c, p2c)
    # Eta contribution #
    # Layer thickness perturbations in Fourier space
    eta1c = result["eta1c"]
    eta2c = result["eta2c"]
    # Momentum flux Jacobian
    tau_jac = mfp.mf_term_Jac(abl)
    # Add eta contributions through the momentum flux
    Fu1 -= np.ravel(tau_jac[0, 4] * eta1c + tau_jac[0, 5] * eta2c)  # Substraction, as the term is brought to the
    Fv1 -= np.ravel(tau_jac[1, 4] * eta1c + tau_jac[1, 5] * eta2c)  # other side, and becomes a forcing term.
    Fu2 -= np.ravel(tau_jac[2, 4] * eta1c + tau_jac[2, 5] * eta2c)
    Fv2 -= np.ravel(tau_jac[3, 4] * eta1c + tau_jac[3, 5] * eta2c)
    # Set up RHS vector #
    rhs = np.concatenate((Fu1, Fv1, Fu2, Fv2))
    return rhs


@njit(parallel=False)
def solve_coupling_system(ks2, ls, U1, V1, U2, V2, nu1, nu2, fc, tau_jac, N2, rhs):
    '''
    Solve the pressure-components of the APM momentum equations in Numba syntax

    Parameters
    ----------
    ks2,ls: numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    fc: float
        Coriolis parameter
    tau_jac: numpy array
        Jacobians of the momentum flux terms
    N2: int
        Total number of grid points
    rhs: numpy array
        Right-hand side

    Returns
    -------
    u_p: 2d numpy array
        Pressure components of the APM velocity perturbation
    '''
    # Re-shape APM state
    RHS = rhs.reshape(4, N2)
    # Set up output
    u_p = np.zeros((4, N2), dtype=np.complex128)
    # Grid size
    Nx2 = ks2.size
    Ny = ls.size
    # Loop over wavenumbers
    for indexk in numba.prange(Nx2):
        for indexl in numba.prange(Ny):
            # Current wavenumber and index #
            k = ks2[indexk]
            l = ls[indexl]
            index = indexl + Ny*indexk
            # Set up matrix for current wavenumber #
            P = np.zeros((4, 4), dtype=np.complex128)
            sigma1 = U1*k+V1*l
            sigma2 = U2*k+V2*l
            # ub1 equation
            P[0, 0] += -1j*sigma1-nu1*(k**2+l**2)
            P[0, 1] += +fc
            # vb1 equation
            P[1, 0] += -fc
            P[1, 1] += -1j*sigma1-nu1*(k**2+l**2)
            # ub2 equation
            P[2, 2] += -1j*sigma2-nu2*(k**2+l**2)
            P[2, 3] += +fc
            # vb2 equation
            P[3, 2] += -fc
            P[3, 3] += -1j*sigma2-nu2*(k**2+l**2)
            # Add vertical momentum flux terms
            P += tau_jac
            # Solve system #
            u_p[:, index] = np.linalg.solve(P, RHS[:, index])
    # Re-shape output
    u_p = u_p.reshape(4*N2)
    return u_p
