"""
Forcing classes for hills and topography in the APM.
"""

__author__ = "Koen Devesse"
__date__ = "January 13, 2022"

import numpy as np
from numba import njit
import numba

from wayve.forcing.apm_forcing import ForcingTerm
from wayve.forcing.pressure_forcing.pressure_forcing import pressure_gradients


class GeneralTopography(ForcingTerm):
    """
    Common class for perturbing topographies.
    """

    def ZOC(self, model):
        """
        Compute the zero-order forcing terms of the APM. This term corresponds to the pressure feedback induced by the
        topography

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
        # Calculate pressure feedback
        N = grid.N2
        p1, p2 = self.pressure_contribution(model, np.zeros(6 * N))
        # Calculate pressure gradients
        Fu1, Fv1, Fu2, Fv2 = pressure_gradients(grid.ks2, grid.ls, p1, p2)
        # Set up RHS vector
        Fe = np.zeros(2*N, dtype=np.complex128)
        F = np.concatenate((Fu1, Fv1, Fu2, Fv2, Fe))
        return F

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
        # Get pressure parametrization
        pressure = model.pressure
        # Get eta
        eta_0 = self.eta_c(grid)
        # Set up pressure coefficients
        Phi = pressure.Phi
        # Calculate pressure fields
        p1, p2 = topography_pressure_contribution(grid.ks2, grid.ls, Phi, eta_0)
        return p1, p2

    def eta(self, grid):
        """
        Return the topography over a given grid
        """
        return np.zeros(grid.shape)

    def eta_c(self, grid):
        """
        Return the Fourier transform of the topography over a given grid
        """
        eta_r = self.eta(grid)
        return grid.r2c(eta_r)


class WitchOfAgnesi(GeneralTopography):
    """
    Class for topographies with a "Witch of Agnesi" shape, which is a ridge parallel to the y-axis.
    """

    def __init__(self, H, L, x_c):
        """
        Initialize a hill with a With of Agnesi shape.

        Parameters
        ----------
        H   float
                Height of the hill
        L   float
                Half-width of the ridge
        x_c float
                Location of the middle of the ridge
        """
        self.__H = H
        self.__L = L
        self.__x_c = x_c

    def eta(self, grid):
        """
        Return the topography over a given grid
        """
        x, y = np.meshgrid(grid.xs, grid.ys, indexing='ij')
        eta_r = self.H / (np.power((x-self.x_c) / self.L, 2) + 1)
        return eta_r

    @property
    def H(self):
        """Height of the ridge"""
        return self.__H

    @property
    def L(self):
        """Half-width of the ridge"""
        return self.__L

    @property
    def x_c(self):
        """Location of the middle of the ridge"""
        return self.__x_c


class GaussianHill(GeneralTopography):
    """
    Class for topographies with a Gaussian shape
    """

    def __init__(self, H, Lx, Ly, x_c, y_c):
        """
        Initialize a hill with a Gaussian shape.

        Parameters
        ----------
        H   float
                Height of the hill
        L   float
                Half-width of the hill
        x_c float
                Location of the middle of the hill
        """
        self.__H = H
        self.__Lx = Lx
        self.__Ly = Ly
        self.__x_c = x_c
        self.__y_c = y_c

    def eta(self, grid):
        """
        Return the topography over a given grid
        """
        x, y = np.meshgrid(grid.xs, grid.ys, indexing='ij')
        eta_r = self.H * np.exp(-0.5*(np.power((x-self.x_c)/self.Lx, 2) + np.power((y-self.y_c)/self.Ly, 2)))
        return eta_r

    @property
    def H(self):
        """Height of the hill"""
        return self.__H

    @property
    def Lx(self):
        """Full width at half maximum in the x-direction of the hill"""
        return self.__Lx

    @property
    def x_c(self):
        """x-coordinate of the middle of the hill"""
        return self.__x_c

    @property
    def Ly(self):
        """Full width at half maximum in the y-direction of the hill"""
        return self.__Ly

    @property
    def y_c(self):
        """y-coordinate of the middle of the ridge"""
        return self.__y_c


@njit(parallel=False)
def topography_pressure_contribution(kv, lv, Phi, eta):
    """
    Compute the pressure contribution of a given topography.
    """
    # Set up arrays
    Nx = len(kv)
    Ny = len(lv)
    p1 = np.zeros((Nx, Ny), dtype=np.complex128)
    p2 = np.zeros((Nx, Ny), dtype=np.complex128)
    # Calculate pressure feedback
    for indexk in numba.prange(Nx):
        for indexl in numba.prange(Ny):
            # Wind farm layer contribution
            p1[indexk, indexl] = Phi[indexk, indexl] * eta[indexk, indexl]
            # Upper layer contribution
            p2[indexk, indexl] = Phi[indexk, indexl] * eta[indexk, indexl]
    return p1, p2



