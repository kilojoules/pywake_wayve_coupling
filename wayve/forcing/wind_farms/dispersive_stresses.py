"""
File containing classes for dispersive stress parametrizations.
"""

__author__ = "Koen Devesse"
__date__ = "April 22, 2024"

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from wayve.forcing.wind_farms.wf_tools import height_average
from wayve.forcing.apm_forcing import ForcingTerm


class DispersiveStresses(ForcingTerm):
    """
    Parametrization for the dispersive stresses in wind farms

    This class uses the wake model velocities on a dense 3D grid to calculate the dispersive stresses. This requires
    that the wind farm associated with the stresses uses a VaryingBackground Coupling that has been instantiated with a
    WakeModelVelocityHandler object (see varying_background.py for additional documentation). This structure then
    ensures that these wake model velocities are calculated when the wind farm is (p)re-processed.
    """

    def __init__(self, wind_farm):
        """
        Initialize this dispersive stress object.
        """
        # Check if WindFarm coupling provides a wake model velocity handler
        if not hasattr(wind_farm.coupling, "wm_velocity_handler") or wind_farm.coupling.wm_velocity_handler is None:
            raise ValueError("Wind farm coupling needs to provide a wake model velocity handler!")
        # WindFarm object
        self.__wind_farm = wind_farm
        # SubGrid stresses
        self.__F_cx = None
        self.__F_cy = None

    @property
    def wind_farm(self):
        """Wind farm associated with the dispersive stresses"""
        return self.__wind_farm

    @property
    def F_cx(self):
        """Dispersive stress forcing in the x-direction"""
        return self.__F_cx

    @F_cx.setter
    def F_cx(self, value):
        self.__F_cx = value

    @property
    def F_cy(self):
        """Dispersive stress forcing in the y-direction"""
        return self.__F_cy

    @F_cy.setter
    def F_cy(self, value):
        self.__F_cy = value

    def preprocess(self, model):
        '''
        Calculate the dispersive stresses.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        '''
        # APM components
        grid = model.grid
        # Get subgrid stresses
        F_cx, F_cy = self.apm_dispersive_stresses(grid, self.wind_farm)
        # Update properties
        self.F_cx = F_cx
        self.F_cy = F_cy

    def reprocess(self, model, result):
        '''
        Calculate the dispersive stresses based on the given APM result.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        result: dict
            State of the APM variables
        '''
        # APM components
        grid = model.grid
        # Get subgrid stresses
        F_cx, F_cy = self.apm_dispersive_stresses(grid, self.wind_farm)
        # Update properties
        self.F_cx = F_cx
        self.F_cy = F_cy

    def ZOC(self, model):
        """
        Compute the zero-order forcing terms of this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zeroth-order contribution terms in Fourier space.
        """
        # APM components
        grid = model.grid
        # Output arrays
        Bu = np.ravel(self.F_cx)
        Bv = np.ravel(self.F_cy)
        return np.concatenate((Bu, Bv, np.zeros((4*grid.N2,), dtype=np.complex128)))

    def apm_dispersive_stresses(self, grid, wind_farm):
        """Calculate the dispersive stress terms of the lower-layer momentum equations"""
        # Get coupling object
        coupling = wind_farm.coupling
        # Get subgrid
        subgrid = coupling.wm_velocity_handler.subgrid
        # Get subgrid velocities
        u_sg = coupling.wm_velocity_handler.u_sg
        v_sg = coupling.wm_velocity_handler.v_sg
        # De-aliasing grid
        grid32 = grid.deal_grid()
        # Get subgrid layer height
        xm, ym = subgrid.xy
        h1_sg = coupling.h1_evaluator(xm, ym)
        # Filtered velocities
        u_f = self.filter(subgrid, u_sg, wind_farm.Lfilter, zero_edge=False)
        v_f = self.filter(subgrid, v_sg, wind_farm.Lfilter, zero_edge=False)
        # Filter difference
        du_d = u_sg - u_f
        dv_d = v_sg - v_f
        # Momentum fluxes #
        uu1_d = np.power(du_d, 2)
        vv1_d = np.power(dv_d, 2)
        uv1_d = np.multiply(du_d, dv_d)
        # Filter
        uu1_d = self.filter(subgrid, uu1_d, wind_farm.Lfilter, zero_edge=True)
        vv1_d = self.filter(subgrid, vv1_d, wind_farm.Lfilter, zero_edge=True)
        uv1_d = self.filter(subgrid, uv1_d, wind_farm.Lfilter, zero_edge=True)
        # Height-average
        z = subgrid.zs
        uu1_d = height_average(uu1_d, h1_sg, z)
        vv1_d = height_average(vv1_d, h1_sg, z)
        uv1_d = height_average(uv1_d, h1_sg, z)
        # Place back on grid
        uu1_d = self.add_sg_to_grid(grid32, subgrid, uu1_d)
        vv1_d = self.add_sg_to_grid(grid32, subgrid, vv1_d)
        uv1_d = self.add_sg_to_grid(grid32, subgrid, uv1_d)
        # Grids for gradient computations
        ks, ls = np.meshgrid(grid.ks2, grid.ls, indexing='ij')
        # Get APM forcing #
        # Dispersive stresses
        Fd_cx = 1.j*ks*grid.r2c_deal(uu1_d) + 1.j*ls*grid.r2c_deal(uv1_d)
        Fd_cy = 1.j*ks*grid.r2c_deal(uv1_d) + 1.j*ls*grid.r2c_deal(vv1_d)
        return Fd_cx, Fd_cy

    def add_sg_to_grid(self, grid, subgrid, sg_field):
        """Interpolate the given 2D field with subgrid resolution onto APM grid"""
        spline = RectBivariateSpline(subgrid.xs, subgrid.ys, sg_field, kx=1, ky=1)
        field = spline(grid.xs, grid.ys)
        return field

    def filter(self, subgrid, sg_field, Lf, zero_edge=False):
        """Apply a Gaussian filter with the given length to the given 2D field with subgrid resolution"""
        # Filter settings
        sigma_x = Lf / (np.sqrt(2.) * subgrid.dx)
        sigma_y = Lf / (np.sqrt(2.) * subgrid.dy)
        sigmas = [sigma_x, sigma_y]
        mode = 'nearest'
        if zero_edge:
            mode = 'constant'
        # Filtered field
        filt = np.zeros(sg_field.shape)
        for i in range(sg_field.shape[2]):
            filt[:, :, i] = gaussian_filter(sg_field[:, :, i], sigmas, mode=mode)
        return filt
