"""
Implementation of the upstream coupling method.
"""

__author__ = "Koen Devesse"
__date__ = "January 4, 2024"

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import VaryingBackground


class Upstream(VaryingBackground):
    """
    A Coupling that generates wake model input based on the velocity upstream of the farm.
    """

    def __init__(self, wake_model, upstr_dist, wm_velocity_evaluator=None):
        """
        Initialize an Upstream Coupling object.

        Parameters
        ----------
        wake_model  WakeModelInterface object
                        Wake model that this Coupling interfaces to.
                        The expected interface is defined in wake_model_interface.py
        upstr_dist  float
                        How far upstream of the first Turbine the velocity is evaluated
        wm_velocity_handler     WakeModelVelocityHandler object (optional)
                        Handler for calculating and storing the wake model velocity fields (default: None)
        """
        # Initialize basic coupling object
        super().__init__(wake_model, wm_velocity_evaluator)
        # Upstream distance
        self.__upstr_dist = upstr_dist

    @property
    def upstr_dist(self):
        """
        How far upstream of the first turbine the velocity is evaluated
        """
        return self.__upstr_dist

    @upstr_dist.setter
    def upstr_dist(self, value):
        self.__upstr_dist = value

    def update_ub_vb(self, model, wind_farm, result):
        """
        Update the evaluator functions for u_b and v_b, based on the given APM state.
        """
        # APM components
        abl = model.abl
        grid = model.grid
        # Get APM perturbation velocities and layer displacement
        du1inf, dv1inf, eta1inf = self.get_upstream_state(wind_farm, abl, grid,
                                                          result['u1r'], result['v1r'], result['eta1r'])
        # Vertical grid and shape function
        z0 = abl.zs[0]
        h1 = abl.H1 + eta1inf
        Nz = 100
        z = np.linspace(z0, h1, Nz, endpoint=True)
        f = self.vertical_profile(abl, z)
        # Height-average shape function
        f_ha = np.trapz(f, x=z) / (z[-1] - z[0])
        # Calculate u_b
        u_b = du1inf / f_ha
        v_b = dv1inf / f_ha
        # Set up evaluator functions
        self.ub_evaluator = lambda x, y: 0. * x + u_b
        self.vb_evaluator = lambda x, y: 0. * x + v_b

    def get_upstream_state(self, windfarm, abl, grid, u1r, v1r, eta1r):
        '''
        Evaluate the perturbation (u1, v1, eta1) upstream of the first turbine.

        Parameters
        ----------
        windfarm: WindFarm object
            wind farm
        abl: ABL object
            atmospheric state
        grid: Grid object
            numerical grid
        u1r, v1r, eta1r: numpy array
            velocity and layer displacement perturbation fields (in real space)

        Returns
        -------
        u1inf,v1inf,eta1inf: float
            perturbation velocity (in x and y) and layer displacement
        '''
        # Get upstream location
        turbines = windfarm.turbines
        WDvector = np.array([abl.U1 / abl.S1, abl.V1 / abl.S1])
        index = type(self).firstTurbine(turbines, WDvector)
        xloc = turbines[index].x - self.upstr_dist * WDvector[0]
        yloc = turbines[index].y - self.upstr_dist * WDvector[1]
        loc = np.array([xloc, yloc])
        # To save time, the interpolation is done only over a 6x6 grid centered in xloc,yloc
        limit = 3
        start_x = int((xloc - grid.xs[0]) / grid.dx - limit)
        end_x = int((xloc - grid.xs[0]) / grid.dx + limit)
        start_y = int((yloc - grid.ys[0]) / grid.dy - limit)
        end_y = int((yloc - grid.ys[0]) / grid.dy + limit)
        x_g = grid.xs[start_x:end_x]
        y_g = grid.ys[start_y:end_y]
        # Set up interpolations functions
        fu = RegularGridInterpolator((x_g, y_g), u1r[start_x:end_x, start_y:end_y])
        fv = RegularGridInterpolator((x_g, y_g), v1r[start_x:end_x, start_y:end_y])
        fe = RegularGridInterpolator((x_g, y_g), eta1r[start_x:end_x, start_y:end_y])
        # Evaluate at upstream location
        u1inf = fu(loc)[0]
        v1inf = fv(loc)[0]
        eta1inf = fe(loc)[0]
        return u1inf, v1inf, eta1inf

    @classmethod
    def firstTurbine(cls, turbines, WDvector):
        '''
        Find first Turbine for a given wind direction

        Parameters
        ----------
        WDvector: numpy array with size 2
            wind direction vector

        Returns
        -------
        _: integer
            index of the first Turbine in the given wind direction
        '''
        Nt = len(turbines)
        # Find first Turbine in a given wind direction by projecting the
        # coordinates on the wind direction vector. The minimal distance
        # corresponds to the first Turbine
        x = np.array([turbines[i].x for i in range(Nt)])
        y = np.array([turbines[i].y for i in range(Nt)])
        coordinates = np.concatenate([x, y]).reshape(Nt, 2, order='F')
        dist = np.dot(coordinates, WDvector)
        return np.argmin(dist)
