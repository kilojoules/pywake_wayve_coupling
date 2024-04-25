"""
Class for pressure-based forcing terms in the APM.
"""

__author__ = "Koen Devesse"
__date__ = "June 13, 2022"

import numpy as np
from numba import njit
from scipy.interpolate import RectBivariateSpline
from scipy.special import erf

from wayve.forcing.apm_forcing import ForcingTerm


class Pressure(ForcingTerm):
    """
    Class of pressure-based forcing terms.
    """

    def __init__(self, p1, p2, xs, ys, dampen=False):
        """
        Initialize this Pressure ForcingTerm object.

        Parameters
        ----------
        p1, p2   2d numpy array
            Pressure field driving the flow
        grid    Grid object
            Grid on which the pressure field is defined
        dampen  Boolean
            Whether or not the pressure gradients are dampened out at the edges of the domain
        """
        self.__p1 = RectBivariateSpline(xs, ys, p1, kx=1, ky=1)
        self.__p2 = RectBivariateSpline(xs, ys, p2, kx=1, ky=1)
        self.__dampen = dampen

    @property
    def p1(self):
        """Spline of the pressure field"""
        return self.__p1

    @property
    def p2(self):
        """Spline of the pressure field"""
        return self.__p2

    @property
    def dampen(self):
        """Whether or not the pressure gradients are dampened out at the edges of the domain"""
        return self.__dampen

    def ZOC(self, model):
        """
        Compute the zero-order forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zero-order perturbation terms in Fourier space.
        """
        # Get grid
        grid = model.grid
        # Get pressure on grid
        p1r = self.p1(grid.xs, grid.ys)
        p2r = self.p2(grid.xs, grid.ys)
        p1c = grid.r2c(p1r)
        p2c = grid.r2c(p2r)
        # Calculate pressure gradients
        Fu1, Fv1, Fu2, Fv2 = pressure_gradients(grid.ks2, grid.ls, p1c, p2c)
        # Dampen pressure forces around domain sides to prevent discontinuities
        if self.dampen:
            for F_c in [Fu1, Fv1, Fu2, Fv2]:
                self.dampen_gradients(grid, F_c)
        # Set up RHS vector
        N = grid.N2
        Fe = np.zeros(2*N, dtype=np.complex128)
        F = np.concatenate((Fu1, Fv1, Fu2, Fv2, Fe))
        return F

    def dampen_gradients(self, grid, F_c):
        """Dampen pressure forces around domain sides to prevent discontinuities"""
        # Dealiasing grid
        grid32 = grid.deal_grid()
        # Grid arrays
        xs = grid32.xs
        ys = grid32.ys
        x, y = np.meshgrid(xs, ys, indexing='ij')
        # Edge region
        d = 50.e3
        x0 = xs[0] + d
        x1 = xs[-1] - d
        y0 = ys[0] + d
        y1 = ys[-1] - d
        # Damping lengthscale
        L_f = 10.e3
        # Damping footprint
        footprint = 1 - (
                ((1 - erf((x - x0)/L_f)) - (1 - erf((x - x1)/L_f))) *
                ((1 - erf((y - y0)/L_f)) - (1 - erf((y - y1)/L_f)))
        ) / 4.
        # IFFT
        F_r = grid.c2r_deal(F_c.reshape(grid.shape2))
        # Apply damping
        F_r = np.multiply(F_r, footprint)
        # FFT
        F_c = np.ravel(grid.r2c_deal(F_r))
        return F_c

    def pressure_contribution(self, model, x):
        """
        Compute the pressure contribution of this forcing object in Fourier space.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Solution vector of the APM

        Returns
        -------
        Two N2 arrays containing the pressure contributions in Fourier space.
        """
        # Get grid
        grid = model.grid
        # Get pressure on grid
        p1r = self.p1(grid.xs, grid.ys)
        p2r = self.p2(grid.xs, grid.ys)
        # Fourier transform
        p1c = grid.r2c(p1r)
        p2c = grid.r2c(p2r)
        return p1c, p2c


@njit(parallel=False)
def pressure_gradients(kv, lv, p1, p2):
    """
    Compute the gradients of the given pressure fields in Fourier space.
    """
    # Set up RHS vector
    Nx = len(kv)
    Ny = len(lv)
    Fu1 = np.zeros((Nx, Ny), dtype=np.complex128)
    Fv1 = np.zeros((Nx, Ny), dtype=np.complex128)
    Fu2 = np.zeros((Nx, Ny), dtype=np.complex128)
    Fv2 = np.zeros((Nx, Ny), dtype=np.complex128)
    # Calculate pressure gradients
    for indexk in range(Nx):
        k = kv[indexk]
        for indexl in range(Ny):
            l = lv[indexl]
            # Wind farm layer contribution
            Fu1[indexk, indexl] = 1j * k * p1[indexk, indexl]
            Fv1[indexk, indexl] = 1j * l * p1[indexk, indexl]
            # Upper layer contribution
            Fu2[indexk, indexl] = 1j * k * p2[indexk, indexl]
            Fv2[indexk, indexl] = 1j * l * p2[indexk, indexl]
    # Combine into output array
    Fu1 = np.ravel(Fu1)
    Fv1 = np.ravel(Fv1)
    Fu2 = np.ravel(Fu2)
    Fv2 = np.ravel(Fv2)
    return Fu1, Fv1, Fu2, Fv2