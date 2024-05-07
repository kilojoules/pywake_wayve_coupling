#!/usr/bin/env python

'''
Implementation of the wake merging method of Lanzilao and Meyers (2021)
'''

__author__ = "Luca Lanzilao, Koen Devesse, Ishaan Sood"
__date__ = "October 15, 2023"

import math

import scipy.linalg
import numba
import numpy as np
from numba import njit

from wayve.forcing.wind_farms.wake_model_coupling.wake_model_interface import UniDirectionalSelfSimilar
from wayve.forcing.wind_farms.wake_model_coupling.wake_models.wake_model_tools import e_spanwise, \
    evaluate_TI, area_circle_segment, area_circle_reflex
from wayve.forcing.forcing_tools import e_streamwise


class UniDirectional(UniDirectionalSelfSimilar):
    """
    An implementation that is based on the unidirectional wake merging method by Lanzilao and Meyers [1]].

    The merging method described there is unidirectional, but the methods still need to output a two-directional flow
    field. To get around this, the flow is decomposed into components in the wind direction, and the components
    perpendicular to it. The wake model is then only applied to the components in the wind direction.

    References
    ----------
    .. [1] Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the presence
    of heterogeneous background velocity fields. Wind Energy, 25(2), 237–259. https://doi.org/10.1002/we.2669
    """

    def __init__(self, ka=0.3837, kb=0.003678, eps_beta=0.2,
                 induction=True, mirrored=True, disk_avg_dir=False, disk_avg_speed=True):
        """
        Initialize a wake merging method object with the given parameters.

        Parameters
        ----------
        ka  float (optional)
                Wake model parameter (default 0.3837, from Niayifar and Porte-Agel (2016))
        kb  float (optional)
                Wake model parameter (default 0.003678, from Niayifar and Porte-Agel (2016))
        eps_beta    float (optional)
                Wake model parameter (default 0.2, from Bastankhah and Porte-Agel (2014))
        induction   Boolean (optional)
                Whether or not an induction model is used (default False, never used in inflow velocity calculations)
        mirrored    Boolean (optional)
                Whether or not turbines are mirrored to emulate deep-array effects (default True)
        disk_avg_dir    Boolean (optional)
                Whether the background flow direction is taken as a disk-average or at hub height (default False)
        disk_avg_speed  Boolean (optional)
                Whether the turbine inflow velocities are calculated as a disk-average or at hub height (default True)
        """
        self.__ka = ka
        self.__kb = kb
        self.__eps_beta = eps_beta
        self.__induction = induction
        self.__mirrored = mirrored
        self.__disk_avg_dir = disk_avg_dir
        self.__disk_avg_speed = disk_avg_speed

    @property
    def ka(self):
        """Wake model parameter"""
        return self.__ka

    @ka.setter
    def ka(self, value):
        self.__ka = value

    @property
    def kb(self):
        """Wake model parameter"""
        return self.__kb

    @kb.setter
    def kb(self, value):
        self.__kb = value

    @property
    def eps_beta(self):
        """Wake model parameter"""
        return self.__eps_beta

    @eps_beta.setter
    def eps_beta(self, value):
        self.__eps_beta = value

    @property
    def induction(self):
        """Whether or not an induction model is used"""
        return self.__induction

    @induction.setter
    def induction(self, value):
        self.__induction = value

    @property
    def mirrored(self):
        """Whether or not turbines are mirrored to emulate deep-array effects"""
        return self.__mirrored

    @mirrored.setter
    def mirrored(self, value):
        self.__mirrored = value

    @property
    def disk_avg_dir(self):
        """Whether the background flow direction is taken as a disk-average or at hub height"""
        return self.__disk_avg_dir

    @disk_avg_dir.setter
    def disk_avg_dir(self, value):
        self.__disk_avg_dir = value

    @property
    def disk_avg_speed(self):
        """Whether the turbine inflow velocities are calculated as a disk-average or at hub height"""
        return self.__disk_avg_speed

    @disk_avg_speed.setter
    def disk_avg_speed(self, value):
        self.__disk_avg_speed = value

    def get_St_Ct_et(self, wind_farm, abl, u_bg_evaluator, apm_evaluator):
        """
        Calculate the turbine inflow velocities (St), the thrust coefficients (Ct), and the turbine directions (et).

        The basic turbine information can be found in the WindFarm object. The unperturbed atmospheric state can be found in
        the ABL object. Additionally, the u_bg_evaluator is a callable that can evaluate the background velocities at
        requested locations.

        Parameters
        ----------
        wind_farm    WindFarm object
            Wind farm for which the calculation is performed
        abl     ABL object
            Information on the unperturbed background state, and atmospheric variables such as TI and z0.
        u_bg_evaluator  callable
            Callable which can evaluate the background velocities at requested locations.
            See VaryingBackground.set_up_u_bg_evaluator in varying_background.py for detailed documentation.
        apm_evaluator   callable
            Callable which can evaluate the layer wind speeds and height at requested locations.
            See VaryingBackground.set_up_apm_evaluators in varying_background.py for detailed documentation.
        """
        # Get evenly spaced points along the rotor disks
        disk_points = self.disk_points(wind_farm, abl)
        # Convert to expected shape
        Nlocs = disk_points.shape[0]*disk_points.shape[1]
        locations = np.reshape(disk_points, (Nlocs, 3))
        # Evaluate the background velocities at the disk points
        velocities = u_bg_evaluator(locations)
        # Convert back to original shape
        disk_velocities = np.reshape(velocities, (disk_points.shape[0], disk_points.shape[1], 2))
        # Calculate disk averages for all turbines
        Ct, St, et = self.inflow_velocities(wind_farm, abl, disk_points, disk_velocities)
        return St, Ct, et

    def background_flow_direction(self, wind_farm, abl):
        """
        Return the direction of the flow according to the wake model, which is assumed to only depend on the unperturbed
        background flow defined in the given ABL object.
        """

        # Get average turbine
        turbines = wind_farm.turbines
        z_h = np.mean([turbine.zh for turbine in turbines])     # Turbine hub height
        D = np.mean([turbine.D for turbine in turbines])        # Turbine diameter

        if self.disk_avg_dir:
            # Get Holoborodko points
            _, zn = holoborodko_points(0., z_h, D)
            # Get wind speeds at Holoborodko points
            u_h = abl.u(zn)         # x-component of the flow
            v_h = abl.v(zn)         # y-component of the flow
            # Get disk-average velocities
            u = np.mean(u_h)
            v = np.mean(v_h)
        else:
            # Wind speed at hub height
            u = abl.u(z_h)
            v = abl.v(z_h)

        # Get wind direction at hub height
        e_str = e_streamwise(u, v)      # Unit vector along the wind direction
        e_span = e_spanwise(u, v)       # Unit vector in cross wind direction

        return e_str, e_span

    def disk_points(self, windfarm, abl):
        """Set up an array of points on the wind farm turbine rotor disks"""

        # Get WindFarm information
        turbines = windfarm.turbines
        Nt = windfarm.Nturb

        # Get Turbine direction (streamwise)
        e_str, e_span = self.background_flow_direction(windfarm, abl)
        theta_str = np.arctan2(e_str[1], e_str[0])

        # Set up points to be scaled for each turbine
        if self.disk_avg_speed:
            # Holoborodko points with unit diameter
            y_disk_yz, z_disk = holoborodko_points(0., 0., 1.)    # Points in yz plane
            n_disk = len(y_disk_yz)
            # Set up evaluation locations (as lists of points for every Turbine)
            disk_points = np.empty((Nt, len(y_disk_yz), 3))
        else:
            # Dimensionless points in yz plane
            y_disk_yz = np.zeros(1)     # y-coordinate at turbine hub
            z_disk = np.ones(1)         # z-coordinate at turbine hub
            # Set up evaluation locations (as lists of points for every Turbine)
            disk_points = np.empty((Nt, 1, 3))

        # Number of points on a rotor disk
        n_disk = disk_points.shape[1]

        # Loop over turbines
        for turb_index, turb in enumerate(turbines):
            # Turbine hub height
            turb_hub = np.array([turb.x, turb.y, turb.zh])
            # Turbine disk locations
            x_disk = - y_disk_yz * np.sin(theta_str)
            y_disk = y_disk_yz * np.cos(theta_str)
            for i in range(n_disk):
                disk_points[turb_index, i, 0] = turb_hub[0] + turb.D * x_disk[i]
                disk_points[turb_index, i, 1] = turb_hub[1] + turb.D * y_disk[i]
                disk_points[turb_index, i, 2] = turb_hub[2] + turb.D * z_disk[i]

        return disk_points

    def inflow_velocities(self, windfarm, abl, disk_points, disk_velocities):
        """Calculate the inflow velocities for all the turbines in the wind farm.

        Parameters
        ----------
        wind_farm    WindFarm object
                        The wind farm, containing Turbine information
        abl         ABL object
                        Information on the atmospheric background conditions
        """
        # Get WindFarm information
        turbines = windfarm.turbines
        Nt = windfarm.Nturb
        xloc = np.array([turbines[k].x for k in range(Nt)])
        yloc = np.array([turbines[k].y for k in range(Nt)])

        # Get number of points on a rotor disk (assumed to be the same for all turbines)
        N_disk = disk_points.shape[1]   # See self.disk_points documentation for more information

        # Get Turbine direction (streamwise)
        e_str, e_span = self.background_flow_direction(windfarm, abl)
        theta_str = np.arctan2(e_str[1], e_str[0])  # Angle in the wind direction

        # Get streamwise velocity at disk points
        disk_s = np.sqrt(np.power(disk_velocities[:, :, 0], 2) + np.power(disk_velocities[:, :, 1], 2))
        disk_theta = np.arctan2(disk_velocities[:, :, 1], disk_velocities[:, :, 0])
        disk_str = disk_s * np.cos(theta_str - disk_theta)

        # Sort turbines along wind direction
        order = self.sort_turbines(windfarm, e_str)

        # Sorted lists
        xloc_sort = np.array([xloc[i] for i in order])
        yloc_sort = np.array([yloc[i] for i in order])
        D_sort = np.array([turbines[i].D for i in order])
        zhs_sort = np.array([turbines[i].zh for i in order])
        disk_str_bg_sort = np.array([disk_str[i, :] for i in order])
        disk_points_sort = np.array([disk_points[i, :, :] for i in order])
        turbines_sort = [turbines[i] for i in order]

        # Background TI
        TI_inf = abl.TI

        # Initial St, Ct, and et estimates
        St_sort = np.array([np.mean(disk_str[i, :]) for i in order])
        Ct_sort = np.array([turbines[i].Ct(np.mean(disk_str[i, :])) for i in order])
        et_sort = np.array([np.array([np.cos(theta_str),
                                      np.sin(theta_str)]) for _ in order])

        # Repeat calculation until Ct converges
        Ct0 = 0.
        step = 0
        max_step = 10
        tol = 1.e-3
        while scipy.linalg.norm(np.abs(Ct_sort - Ct0)) / Nt > tol and step < max_step:
            # Increase step
            step += 1
            Ct0 = np.copy(Ct_sort)
            # Evaluate TI at Turbine locations
            TI = evaluate_TI(Nt, e_str, e_span, order, xloc_sort, yloc_sort, D_sort, Ct_sort, TI_inf,
                             self.ka, self.kb)
            TI_sort = np.array([TI[i] for i in order])
            # Apply wakes for all turbines #
            # Initialize velocities as background velocities
            disk_str_sort = disk_str_bg_sort
            # Flatten arrays of locations and velocities
            disk_str_flat = np.reshape(disk_str_sort, (Nt*N_disk))
            locations = np.reshape(disk_points_sort, (Nt*N_disk, 3))
            for i in range(Nt):
                # Get inflow conditions for current Turbine
                St_sort[i] = np.mean(disk_str_sort[i, :])
                et_sort[i] = np.array([np.cos(theta_str), np.sin(theta_str)])
                Ct_sort[i] = turbines_sort[i].Ct(St_sort[i])
                if Ct_sort[i] != 0.:
                    # Get wake of current Turbine
                    W = gaussian_wake_function(locations, TI_sort[i], Ct_sort[i], xloc_sort[i], yloc_sort[i], D_sort[i],
                                               zhs_sort[i], theta_str, ka=self.ka, kb=self.kb, eps_beta=self.eps_beta,
                                               mirr=self.mirrored)
                    # Apply wake of current Turbine
                    disk_str_flat = np.multiply(disk_str_flat, 1. - W)
                    disk_str_sort = np.reshape(disk_str_flat, (Nt, N_disk))

        # Unsort turbines
        inv_order = np.argsort(order)
        Ct = Ct_sort[inv_order]
        St = St_sort[inv_order]
        et = et_sort[inv_order]

        return Ct, St, et

    def wake_deficit(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Product over all wake functions (1-W_k), calculated on the given subgrid.

        The basic turbine information can be found in the WindFarm object. The unperturbed atmospheric state can be found in
        the ABL object. Additionally, the u_bg_evaluator is a callable that can evaluate the background velocities at
        requested locations.

        Parameters
        ----------
        wind_farm    WindFarm object
            Wind farm for which the calculation is performed
        abl     ABL object
            Information on the unperturbed background state, and atmospheric variables such as TI and z0.
        u_bg_evaluator  callable
            Callable which can evaluate the background velocities at requested locations.
            See VaryingBackground.set_up_u_bg_evaluator in varying_background.py for detailed documentation.
        apm_evaluator   callable
            Callable which can evaluate the layer wind speeds and height at requested locations.
            See VaryingBackground.set_up_apm_evaluators in varying_background.py for detailed documentation.
        subgrid     SubGrid object
            Grid on which the velocity should be evaluated
        """

        # Get the turbine thrust coefficients
        _, Ct, _ = self.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, apm_evaluator)

        # Get grid information
        Nx = subgrid.Nx
        Ny = subgrid.Ny
        Nz = subgrid.Nz

        # Ambient TI
        TI_inf = abl.TI

        # Background flow direction
        e_str, e_span = self.background_flow_direction(wind_farm, abl)

        # Get wind farm information
        turbines = wind_farm.turbines
        Nt = wind_farm.Nturb

        # Set up output array
        wake_deficit = np.ones((Nx, Ny, Nz))

        # Turbine locations w.r.t. subgrid
        xloc = np.array([turbines[k].x - subgrid.x_min for k in range(Nt)])
        yloc = np.array([turbines[k].y - subgrid.y_min for k in range(Nt)])

        # Sort turbines along wind direction - for TI
        order = self.sort_turbines(wind_farm, e_str)
        xloc_sort = np.array([xloc[i] for i in order])
        yloc_sort = np.array([yloc[i] for i in order])
        D_sort = np.array([turbines[i].D for i in order])
        Ct_sort = Ct[order]
        theta = np.arctan2(e_str[1], e_str[0])      # Angle in the wind direction

        # Evaluate TI at Turbine locations
        TI = evaluate_TI(Nt, e_str, e_span, order, xloc_sort, yloc_sort, D_sort, Ct_sort, TI_inf,
                         self.ka, self.kb)

        # Set up meshgrid
        Xs, Ys = subgrid.xy
        Xs -= subgrid.x_min
        Ys -= subgrid.y_min
        z = subgrid.zs

        for turb in range(Nt):
            if Ct[turb] != 0.:
                wake_deficit = wake_deficit * (1 -
                                               type(self).wake_function(Nx, Ny, Nz, Xs, Ys, z, TI[turb], Ct[turb],
                                                                        xloc[turb], yloc[turb], turbines[turb].D,
                                                                        turbines[turb].zh, theta, ka=self.ka,
                                                                        kb=self.kb, ind=self.induction,
                                                                        mirr=self.mirrored)
                                               )
        return wake_deficit

    def sort_turbines(self, windfarm, e_str):
        """Sort turbines along wind direction"""
        # Get wind farm information
        turbines = windfarm.turbines
        Nt = windfarm.Nturb
        # Turbine locations
        xloc = np.array([turbines[k].x for k in range(Nt)])
        yloc = np.array([turbines[k].y for k in range(Nt)])
        # Sort turbines along wind direction
        coordinates = np.concatenate([xloc, yloc]).reshape(Nt, 2, order='F')
        dist = np.dot(coordinates, e_str)
        order = np.argsort(dist)
        return order

    @classmethod
    def wake_function(cls, Nx,Ny,Nz,Xs,Ys,z,TI,Ct,xloc,yloc,D,zh,theta,ka=0.3837,kb=0.003678,eps_beta=0.2,
                      ind=False,mirr=True):
        '''
        Evaluate the wake deficit function in (xloc,yloc)
        '''
        W = gaussian_wake_function_grid(Nx, Ny, Nz, Xs, Ys, z, TI, Ct, xloc, yloc, D, zh, theta, ka=ka, kb=kb,
                                        eps_beta=eps_beta, ind=ind, mirr=mirr)
        return W

    def xy_plane(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, xs, ys, z):
        """
        Compute an xy cross-section of the flow.
        """

        # Get the turbine thrust coefficients
        _, Ct, _ = self.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, apm_evaluator)

        # Get grid information
        Nx = len(xs)
        Ny = len(ys)
        Nz = 1

        # Ambient TI
        TI_inf = abl.TI

        # Background flow direction
        e_str, e_span = self.background_flow_direction(wind_farm, abl)

        # Get wind farm information
        turbines = wind_farm.turbines
        Nt = wind_farm.Nturb

        # Set up output array
        wake_deficit = np.ones((Nx, Ny, Nz))

        # Turbine locations
        xloc = wind_farm.xs
        yloc = wind_farm.ys

        # Sort turbines along wind direction - for TI
        order = self.sort_turbines(wind_farm, e_str)
        xloc_sort = np.array([xloc[i] for i in order])
        yloc_sort = np.array([yloc[i] for i in order])
        D_sort = np.array([turbines[i].D for i in order])
        Ct_sort = Ct[order]
        theta = np.arctan2(e_str[1], e_str[0])      # Angle in the wind direction

        # Evaluate TI at Turbine locations
        TI = evaluate_TI(Nt, e_str, e_span, order, xloc_sort, yloc_sort, D_sort, Ct_sort, TI_inf,
                         self.ka, self.kb)

        # Set up meshgrid
        Xs, Ys = np.meshgrid(xs, ys, indexing='ij')
        zs = np.array([z])
        x_m, y_m, z_m = np.meshgrid(xs, ys, zs, indexing="ij")

        for turb in range(Nt):
            if Ct[turb] != 0.:
                wake_deficit = wake_deficit * (1 -
                                               type(self).wake_function(Nx, Ny, Nz, Xs, Ys, zs, TI[turb], Ct[turb],
                                                                        xloc[turb], yloc[turb], turbines[turb].D,
                                                                        turbines[turb].zh, theta, ka=self.ka,
                                                                        kb=self.kb, ind=self.induction,
                                                                        mirr=self.mirrored)
                                               )
        # Get background velocities #
        # SubGrid coordinates
        x_locs = np.ravel(x_m)
        y_locs = np.ravel(y_m)
        z_locs = np.ravel(z_m)
        locations = np.stack([x_locs, y_locs, z_locs], axis=1)
        # Evaluate background velocities
        vel_bg = u_bg_evaluator(locations)
        # Reshape into grid
        u_bg = np.reshape(vel_bg[:, 0], (Nx, Ny, Nz))
        v_bg = np.reshape(vel_bg[:, 1], (Nx, Ny, Nz))
        # Split up into streamwise and spanwise components
        str_bg = u_bg * e_str[0] + v_bg * e_str[1]
        span_bg = u_bg * e_span[0] + v_bg * e_span[1]
        # Add wakes
        str_wm = np.multiply(str_bg, wake_deficit)
        # Convert to u and v components
        u_wm = str_wm * e_str[0] + span_bg * e_span[0]
        v_wm = str_wm * e_str[1] + span_bg * e_span[1]
        # Get 2D arrays
        u_bg = u_bg[:, :, 0]
        v_bg = v_bg[:, :, 0]
        u_wm = u_wm[:, :, 0]
        v_wm = v_wm[:, :, 0]
        return u_bg, v_bg, u_wm, v_wm


@njit(parallel=True)
def gaussian_wake_function_grid(Nx, Ny, Nz, Xs, Ys, z, TI, Ct, xloc, yloc, D, zh, theta,
                                ka=0.3837, kb=0.003678, eps_beta=0.2,
                                ind=False, mirr=True):
    '''
    Evaluate the wake deficit function over a 3D grid in Numba syntax.

    This method uses a regularly spaced 3D grid to speed up the implementation. Both the Gaussian wake model and the
    self-similar induction model are applied.
    '''
    # Output array
    W = np.zeros((Nx,Ny,Nz), dtype=np.float64)
    # Get x and y coordinates in Turbine reference frame
    Xwake = Xs - xloc   # x=0 at xloc
    Ywake = Ys - yloc   # y=0 at yloc
    Xwake_rot = Xwake*np.cos(theta) + Ywake*np.sin(theta)   # axis rotation
    Ywake_rot = -Xwake*np.sin(theta) + Ywake*np.cos(theta)  # axis rotation
    # Wake parameters
    kwake = ka*TI+kb
    # Get sigma at locations
    beta = 0.5*(1+np.sqrt(1-Ct))/np.sqrt(1-Ct)
    eps = eps_beta*np.sqrt(beta)
    sigma_y_d0 = kwake * Xwake_rot / D + eps
    sigma_z_d0 = sigma_y_d0
    sigma_d0_sq = sigma_y_d0 * sigma_z_d0
    # Ct modified according to Zong&Porte-agel (2020)
    Cts = np.empty((Nx,Ny))
    for i in numba.prange(Nx):
        for j in numba.prange(Ny):
            Cts[i,j] = Ct*(1+math.erf(max(0., Xwake_rot[i, j])/D))/2
            if Cts[i,j]==Ct/2:
                Cts[i, j] = 0
    C_erf = 1 - np.sqrt(1-np.minimum(np.ones((Nx,Ny)), Cts/(8*sigma_d0_sq)))
    # Yaw displacement
    deltas = np.zeros((Nx,Ny))
    # 3D field
    for i in numba.prange(Nx):
        for j in numba.prange(Ny):
            if Xwake_rot[i, j] > 0.:
                for k in numba.prange(Nz):
                    spread = np.exp(-(((z[k]-zh)/sigma_z_d0[i,j])**2 +
                                      (Ywake_rot[i,j]/sigma_y_d0[i,j])**2)
                                    / (2*D**2))
                    if mirr:
                        spread += np.exp(-(((z[k]+zh)/sigma_z_d0[i,j])**2 +
                                      (Ywake_rot[i,j]/sigma_y_d0[i,j])**2)
                                    / (2*D**2))
                    W[i,j,k] = C_erf[i,j] * spread

    # Induction region
    if ind:
        # Parameters
        R = D / 2
        alpha = 8/9
        beta = np.sqrt(2)
        gamma = min(1, 1/Ct)
        # TFM values
        lambd = 0.587
        eta = 1.32
        # Axial induction factor
        a_0 = 0.5 * (1 - np.sqrt(1 - gamma * Ct))
        # Scaled distance
        x_nd = Xwake_rot / R
        # Loop
        for i in numba.prange(Nx):
            for j in numba.prange(Ny):
                if Xwake_rot[i, j] <= 0.:
                    # Centerline induction
                    a = a_0 * (1 + x_nd[i, j] / np.sqrt(1 + np.power(x_nd[i, j], 2)))
                    # Non-dimensional half-width
                    r_12 = R * np.sqrt(lambd * (eta + np.power(x_nd[i, j], 2)))
                    for k in numba.prange(Nz):
                        # Induction zone shape
                        r = np.sqrt((z[k]-zh)**2+Ywake_rot[i,j]**2)
                        eps = r / r_12
                        f = np.power(np.cosh(beta * eps), -alpha)
                        if mirr:
                            r_m = np.sqrt((z[k]+zh)**2+Ywake_rot[i,j]**2)
                            eps_m = r_m / r_12
                            f += np.power(np.cosh(beta * eps_m), -alpha)
                        W[i, j, k] = a * f

    return W


@njit(parallel=False)
def gaussian_wake_function(locations,TI,Ct,xloc,yloc,D,zh,theta,ka=0.3837,kb=0.003678,eps_beta=0.2,mirr=True):
    '''
    Evaluate the Gaussian wake deficit function of a turbine in (xloc,yloc) in the given locations
    '''
    # Get number of locations
    N_loc = locations.shape[0]
    # Get x and y coordinates in Turbine reference frame
    ref_locations = np.empty(locations.shape)
    for i in numba.prange(N_loc):
        # Get coordinates in new reference frame
        Xwake = locations[i, 0] - xloc  # x=0 at xloc
        Ywake = locations[i, 1] - yloc  # y=0 at yloc
        ref_locations[i, 0] = Xwake*np.cos(theta) + Ywake*np.sin(theta)     # axis rotation
        ref_locations[i, 1] = -Xwake*np.sin(theta) + Ywake*np.cos(theta)    # axis rotation
        ref_locations[i, 2] = locations[i, 2]   # z-coordinate unchanged
        # Set x<0 to x=0 so that C_erf=0 for x<0
        ref_locations[i, 0] = max(0., ref_locations[i, 0])
    # Wake parameters
    kwake = ka*TI+kb
    # Near-wake yaw effects
    kwake_alpha = 2.32
    kwake_beta = 0.154
    # Get sigma at locations
    beta = 0.5*(1+np.sqrt(1-Ct))/np.sqrt(1-Ct)
    eps = eps_beta*np.sqrt(beta)
    sigma_y_d0 = kwake * ref_locations[:, 0] / D + eps
    sigma_z_d0 = sigma_y_d0
    sigma_d0_sq = sigma_y_d0 * sigma_z_d0
    # Ct modified according to Zong&Porte-agel (2020)
    Cts = np.empty(N_loc)
    for i in numba.prange(N_loc):
        Cts[i] = Ct * (1 + math.erf(ref_locations[i, 0]/D)) / 2
        if Cts[i] == Ct/2:
            Cts[i] = 0
    C_erf = 1 - np.sqrt(1-np.minimum(np.ones(N_loc), Cts/(8*sigma_d0_sq)))
    # Evaluate shape function at locations
    spread = np.exp(-(np.divide((ref_locations[:, 2]-zh), sigma_z_d0)**2 +
                      np.divide(ref_locations[:, 1], sigma_y_d0)**2)
                    / (2*D**2))
    if mirr:
        spread += np.exp(-(np.divide((ref_locations[:, 2]+zh), sigma_z_d0)**2 +
                           np.divide(ref_locations[:, 1], sigma_y_d0)**2)
                         / (2*D**2))
    W = C_erf * spread
    return W


def holoborodko_points(y_c, z_c, D):
    """
    Sets up 16 quadrature points on a disk in the y-z plane according to the Holoborodko method.

    Parameters
    ----------
    y_h     float
        y-coordinate of the center
    z_h     float
        z-coordinate of the center
    D       float
        Diameter of the circle
    """
    # Number of quadrature points
    Nq = 16
    # Set up coordinate arrays
    yn = np.zeros(Nq)
    zn = np.zeros(Nq)
    # Loop over quadrature points
    for n in numba.prange(Nq):
        phi = 2 * np.pi * n / Nq
        r = np.sqrt((3. + (1. - 2 * (n % 2)) * np.sqrt(3.)) / 6.) * D / 2.0
        yn[n] = y_c + r * np.cos(phi)
        zn[n] = z_c + r * np.sin(phi)
    return yn, zn


def holoborodko_points_evenly_spaced(y_c, z_c, D, Nr=6):
    """
    Sets up evenly spaced points on a disk in the y-z plane according to the Holoborodko method.

    Parameters
    ----------
    y_h     float
        y-coordinate of the center
    z_h     float
        z-coordinate of the center
    D       float
        Diameter of the circle
    Nr      int (optional)
        Number of radial circles (default=6)
    """
    yn = np.array([y_c])
    zn = np.array([z_c])
    k = 1
    for r in np.linspace(D/(2.*Nr), D/2., Nr, endpoint=True):
        n = round(np.pi / np.arcsin(1/(2*k)))   # Arc-wise spacing should be roughly equal to radial spacing
        phi = np.linspace(0., 2*np.pi, n+1)
        yn = np.concatenate((yn, y_c + r * np.cos(phi)), axis=None)
        zn = np.concatenate((zn, z_c + r * np.sin(phi)), axis=None)
        k += 1
    return yn, zn
