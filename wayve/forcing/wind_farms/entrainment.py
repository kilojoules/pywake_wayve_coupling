"""
Parametrizations for momentum entrainment above wind farms in the APM
"""

__author__ = "Koen Devesse"
__date__ = "April 8, 2024"

import numpy as np

from wayve.forcing.apm_forcing import ForcingTerm
from wayve.forcing.forcing_tools import fill_shape, e_streamwise
from wayve.forcing.wind_farms.wf_tools import filter_2d_scipy


class WFMF(ForcingTerm):
    """Class for the added turbulent momentum flux caused by a wind farm"""

    def __init__(self, wind_farm):
        '''
        Initialise the entrainment parametrization for the given wind farm

        Parameters
        ----------
        wind_farm    WindFarm object
            the farm causing the added momentum flux
        '''
        self.__wind_farm = wind_farm

    def ZOC(self, model):
        """
        Compute the zero-order forcing terms of the APM.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zero-order perturbation terms.
        """
        H1 = model.abl.H1
        H2 = model.abl.H2
        Nx = model.grid.Nx2
        Ny = model.grid.Ny

        # Add momentum flux
        tau0x, tau0y = self.MF0(self.wind_farm, model.abl, model.grid.deal_grid())
        Tu1 = np.ravel(model.grid.r2c_deal(tau0x))/H1
        Tv1 = np.ravel(model.grid.r2c_deal(tau0y))/H1
        Tu2 = -np.ravel(model.grid.r2c_deal(tau0x))/H2
        Tv2 = -np.ravel(model.grid.r2c_deal(tau0y))/H2

        return np.concatenate((Tu1,Tv1,Tu2,Tv2,np.zeros((2*Nx*Ny,),dtype=np.complex128)))

    def FOC(self, model, x):
        """
        Compute the first-order contributions of this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Perturbation array, containing u1, v1, u2, v2, eta1, and eta2.
        Returns
        -------
        A 6*N2 array containing the first-order contribution terms in Fourier space.
        """
        # Output array
        F = 0. * x
        # Grid size
        N = model.grid.N2
        # Layer thicknesses
        H1 = model.abl.H1
        H2 = model.abl.H2
        # Change in tau #
        tau1x, tau1y = self.MF1(self.wind_farm, model.abl, model.grid.deal_grid(), x)
        F[0:N] += np.ravel(model.grid.r2c_deal(tau1x / H1))
        F[N:2*N] += np.ravel(model.grid.r2c_deal(tau1y / H1))
        F[2*N:3*N] += -np.ravel(model.grid.r2c_deal(tau1x / H2))
        F[3*N:4*N] += -np.ravel(model.grid.r2c_deal(tau1y / H2))
        # Change in H #
        # Extract dependent variables eta1 and eta2 from the vector
        eta1c = x[4*N:5*N]
        eta2c = x[5*N:]
        # Convert to real space
        eta1r = model.grid.c2r_deal(eta1c.reshape(model.grid.shape2))
        eta2r = model.grid.c2r_deal(eta2c.reshape(model.grid.shape2))
        # Add momentum flux
        tau0x, tau0y = self.MF0(self.wind_farm, model.abl, model.grid.deal_grid())
        F[0:N] += -np.ravel(model.grid.r2c_deal(np.multiply(tau0x, eta1r)))/H1**2
        F[N:2*N] += -np.ravel(model.grid.r2c_deal(np.multiply(tau0y, eta1r)))/H1**2
        F[2*N:3*N] += np.ravel(model.grid.r2c_deal(np.multiply(tau0x, eta2r)))/H2**2
        F[3*N:4*N] += np.ravel(model.grid.r2c_deal(np.multiply(tau0y, eta2r)))/H2**2
        return F

    def MF0(self, wind_farm, abl, grid32):
        """
        Zeroth-order contribution of the added momentum flux

        Default implementation returns zero arrays.
        """
        Nx = grid32.Nx
        Ny = grid32.Ny
        # Output array
        tau = np.zeros((2, Nx, Ny))
        return tau[0], tau[1]

    def MF1(self, wind_farm, abl, grid32, x):
        """
        First-order contribution of the added momentum flux

        Default implementation returns zero arrays.
        """
        Nx = grid32.Nx
        Ny = grid32.Ny
        # Output array
        tau = np.zeros((2, Nx, Ny))
        return tau[0], tau[1]

    def MFtot(self, wind_farm, abl, grid32, x):
        """
        Total added momentum flux

        Default implementation returns the sum of MF0 and MF1.
        """
        tau0x, tau0y = self.MF0(wind_farm, abl, grid32)
        tau1x, tau1y = self.MF1(wind_farm, abl, grid32, x)
        return tau0x + tau1x, tau0y + tau1y

    @property
    def wind_farm(self):
        """Wind farm associated with this increased momentum flux"""
        return self.__wind_farm


class ConstantFlux(WFMF):
    """
    Class for WindFarm-induced momentum flux that assumes a constant added flux in the shape of the farm, shifted downstream.
    """

    def __init__(self, wind_farm, a=0.120, d=27.8):
        """
        Initialize this momentum flux object.
        """
        # Basic initialization
        super().__init__(wind_farm)
        # Scaling coefficients
        self.__a = a
        self.__d = d

    @property
    def a(self):
        """Strength of the added momentum flux, scaled with the WindFarm force density"""
        return self.__a

    @a.setter
    def a(self, value):
        self.__a = value

    @property
    def d(self):
        """Downstream shift of the added momentum flux, scaled with the turbine diameter"""
        return self.__d

    @d.setter
    def d(self, value):
        self.__d = value

    @staticmethod
    def scale(wind_farm, abl):
        area = wind_farm.area
        Ct_avg = np.mean([wind_farm.coupling.Ct])
        ra_avg = np.mean([turbine.rotorarea for turbine in wind_farm.turbines])
        Ct = wind_farm.Nturb*Ct_avg*ra_avg / area
        return 0.5*Ct*abl.S1**2

    def step(self, wind_farm, abl):
        scale = ConstantFlux.scale(wind_farm, abl)
        return self.a * scale

    def delay(self, wind_farm, abl):
        D = np.mean([turbine.D for turbine in wind_farm.turbines])
        return self.d * D

    def MF0(self, wind_farm, abl, grid32):
        # Added momentum flux
        footprint = self.footprint(wind_farm, abl, grid32)
        dtau_wf = - self.step(wind_farm, abl) * footprint
        # x and y components
        e_t = e_streamwise(abl.U1, abl.V1)
        dtau_x = dtau_wf * e_t[0]
        dtau_y = dtau_wf * e_t[1]
        return dtau_x, dtau_y

    def footprint(self, wind_farm, abl, grid32):
        """
        Footprint defining the shape and location of the added momentum flux.
        """
        # Delay due to IBL growth #
        e_t = e_streamwise(abl.U1, abl.V1)
        delay = self.delay(wind_farm, abl)
        # Set up subgrid #
        # Spacing
        Lf = wind_farm.Lfilter
        dx = Lf / 10
        # Domain boundaries
        x_min = wind_farm.xstart - 10 * Lf + delay * e_t[0]
        y_min = wind_farm.ystart - 10 * Lf + delay * e_t[1]
        x_max = wind_farm.xend + 10 * Lf + delay * e_t[0]
        y_max = wind_farm.yend + 10 * Lf + delay * e_t[1]
        # Grid arrays
        xs = np.arange(x_min, x_max, dx)
        ys = np.arange(y_min, y_max, dx)
        xm, ym = np.meshgrid(xs, ys, indexing="ij")
        # Footprint calculation #
        # Get unfiltered footprint on subgrid
        footprint_sg = fill_shape(wind_farm.vertices, xm - delay * e_t[0], ym - delay * e_t[1])
        # Filter and place footprint on APM grid
        footprint = filter_2d_scipy(footprint_sg, grid32.xs, grid32.ys, xs, ys, Lf, zero_edge=True)
        return footprint
