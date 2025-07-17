'''
A file containing pressure closure equations based on linear gravity wave theory.
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "June 1, 2023"

from cmath import sqrt

import numba
import numpy as np
from numba import njit
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from wayve.pressure.gravity_waves import nonuniform_methods as nus
from wayve.pressure.pressure_parametrizations import PressureParametrization


class GravityWaves(PressureParametrization):
    """A generic class for the pressure parametrizations based on linear gravity wave theory."""

    def __init__(self, dynamic=True, rotating=False):
        """
        Initialize a Uniform PressureParametrization with the given settings.

        The stratification coefficients are initialized as None.

        Parameters
        ----------
        dynamic    bool
            Whether the internal gravity waves are treated as being hydrodynamic or hydrostatic
        rotating    bool
            Whether the internal gravity waves are treated as being rotating or non-rotating
        """
        super().__init__()
        self.__dynamic = dynamic
        self.__rotating = rotating

    def AResonanceCalculator(self, abl, shape, ks, ls):
        '''
        Compute the internal wave resonance parameter A

        Returns
        -------
        A: 2d numpy array
            internal wave resonance parameter
        '''
        raise Exception("IGW resonance calculation not implemented yet for this class!")

    def get_waves(self, abl, kv, lv):
        """
        Return the wave information for a given k and l vector.
        The returned amplitudes are those of the up- and down-going components of the vertical velocity perturbation.

        Parameters
        ----------
        kv  Wave-numbers in the x-direction
        lv  Wave-numbers in the y-direction

        Returns
        -------
        m_squared   Square of the vertical wave-numbers
        amplitudes  Amplitudes of the up- and down-going waves (amplitudes of vertical velocity)
        """
        raise Exception("IGW amplitude calculation not implemented yet for this class!")

    def pcm_input_setup(self, abl):
        """Set up the inputs for the piece-wise constant method"""
        raise Exception("PCM input setup not implemented yet for this class!")

    @property
    def dynamic(self):
        return self.__dynamic

    @property
    def rotating(self):
        return self.__rotating


class Uniform(GravityWaves):
    """A class for the GW pressure parametrization for uniformly stratified upper atmospheres."""

    def get_Phi(self, abl, grid):
        """
        Compute the stratification coefficients for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        return self.evaluate_Phi(grid.shape2, grid.ks2, grid.ls,
                                 abl.gprime, abl.U3, abl.V3, abl.N, abl.fc,
                                 self.dynamic, self.rotating)

    def AResonanceCalculator(self, abl, shape, ks, ls):
        '''
        Compute the internal wave resonance parameter A

        Returns
        -------
        A: 2d numpy array
            internal wave resonance parameter
        '''
        return np.ones(shape)

    def get_waves(self, abl, kv, lv):
        """
        Return the wave information for a given k and l vector.
        The returned amplitudes are those of the up- and down-going components of the vertical velocity perturbation.

        Parameters
        ----------
        kv  Wave-numbers in the x-direction
        lv  Wave-numbers in the y-direction

        Returns
        -------
        m_squared   Square of the vertical wave-numbers
        amplitudes  Amplitudes of the up- and down-going waves (amplitudes of vertical velocity)
        """
        Nx = len(kv)
        Ny = len(lv)
        amplitudes = np.zeros([Nx, Ny, 2], dtype=np.complex128)
        m_squared = np.zeros([Nx, Ny, 1], dtype=np.complex128)
        for indexk, k in enumerate(kv):
            for indexl, l in enumerate(lv):
                Omega3 = -(abl.U3 * k + abl.V3 * l)
                if not Omega3 == 0:
                    # Dispersion relation
                    nominator = -abl.N ** 2
                    if self.dynamic:
                        nominator += Omega3 ** 2
                    denominator = Omega3 ** 2
                    if self.rotating:
                        denominator -= abl.fc ** 2
                    m2 = - (k**2 + l**2) * nominator / denominator
                    m_squared[indexk, indexl] = [m2]
                    # Up- or downgoing waves
                    if m2 < 0.:  # Evanescent
                        amplitudes[indexk, indexl, 0] = 1.
                    else:  # Propagating
                        if np.sign(Omega3) < 0:
                            amplitudes[indexk, indexl, 0] = 1.
                        else:
                            amplitudes[indexk, indexl, 1] = 1.
        h_interfaces = np.array([abl.H])
        return h_interfaces, m_squared, amplitudes

    def pcm_input_setup(self, abl):
        """Set up the inputs for the piece-wise constant method.

        Uniform atmospheres can be considered to have only one sublayer.
        """
        # Sublayer parameters
        N_param = np.array([abl.N])
        u_param = np.array([abl.U3])
        v_param = np.array([abl.V3])
        d2u_param = np.array([0.])
        d2v_param = np.array([0.])
        # Sublayer interface height
        h_interfaces = np.array([abl.H])
        # Sublayer interface parameters
        u_int = np.array([abl.U3])
        v_int = np.array([abl.V3])
        du_int = np.array([0.])
        dv_int = np.array([0.])
        return N_param, u_param, v_param, d2u_param, d2v_param, h_interfaces, u_int, v_int, du_int, dv_int

    @staticmethod
    @njit(parallel=False)
    def evaluate_Phi(shape, ks, ls, gprime, U3, V3, N, fc, dynamic, rotating):
        '''
        Compute complex stratification coefficients Phi in Numba syntax

        Parameters
        ----------
        shape: numpy array
            Dimensions of the numerical domain
        ks,ls: numpy array
            Array with wave numbers in dimension 0 (Fourier space)
            Array with wave numbers in dimension 1 (Fourier space)
        gprime: float
            Reduced gravity
        U3: float
            Velocity in the free atmosphere in dimension 0
        V3: float
            Velocity in the free atmosphere in dimension 1
        N: float
            Brunt-Vaisala frequency
        fc: float
            Coriolis parameter
        dynamic: Boolean
            Whether the upper atmosphere is considered hydrodynamic or not
        rotating: Boolean
            Whether the upper atmosphere is considered rotating or non-rotating

        Returns
        -------
        PHI: 2d numpy array
            complex stratification coefficient
        '''
        # Inversion layer waves
        Phi = gprime * np.ones(shape, dtype=np.complex128)
        # Internal gravity waves
        for indexk in numba.prange(ks.size):
            for indexl in numba.prange(ls.size):
                k = ks[indexk]
                l = ls[indexl]
                Omega3 = -(U3 * k + V3 * l)
                if not Omega3 == 0:
                    # Dispersion relation
                    nominator = -N ** 2
                    if dynamic:
                        nominator += Omega3 ** 2
                    denominator = Omega3 ** 2
                    if rotating:
                        denominator -= fc ** 2
                    m2 = - (k**2 + l**2) * nominator / denominator
                    # Up- or downgoing waves
                    if m2 < 0.:  # Evanescent
                        m = sqrt(m2)
                    else:  # Propagating
                        m = - np.sign(Omega3) * sqrt(m2)
                    # Stratification coefficient
                    if dynamic:
                        Phi[indexk, indexl] += 1j / m * (N ** 2 - Omega3 ** 2)
                    else:
                        Phi[indexk, indexl] += 1j / m * (N ** 2)
        return Phi


class NonUniform(GravityWaves):
    """A class for the GW pressure parametrization for upper atmospheres with varying stratification and wind speed."""

    def __init__(self, n_layers=-1, order=1, dynamic=True):
        """
        Initialize a NonUniform GW PressureParametrization with the given settings.

        Rotating gravity waves are not implemented for this class.

        Parameters
        ----------
        n_layers    int
            The number of sublayers used by the gravity wave model
            If n_layers <= 0, the vertical grid of the ABL (zs) is used to set up the sublayers.
        order       int
            The order of the splines used to set up the velocity profiles
        dynamic    bool
            Whether the internal gravity waves are treated as being hydrodynamic or hydrostatic
        """
        super().__init__(dynamic, False)
        self.__n_layers = n_layers
        self.__order = order

    def get_Phi(self, abl, grid):
        """
        Compute the stratification coefficients for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        # Get piecewise method inputs
        N_param, u_param, v_param, d2u_param, d2v_param, h_int, u_int, v_int, du_int, dv_int = self.pcm_input_setup(abl)
        # Evaluate Phi
        return self.evaluate_Phi(grid.shape2, grid.ks2, grid.ls,
                                 abl.gprime,
                                 N_param, u_param, v_param, d2u_param, d2v_param, h_int, u_int, v_int, du_int, dv_int,
                                 self.dynamic)

    def AResonanceCalculator(self, abl, shape, ks, ls):
        '''
        Compute the internal wave resonance parameter A

        Returns
        -------
        A: 2d numpy array
            internal wave resonance parameter
        '''
        if not self.dynamic:
            raise Exception("Hydrostatic upper atmosphere not implemented!")
        if self.rotating:
            raise Exception("Rotating upper atmosphere not implemented!")

        def energyLayer(m2_sl, N, W_plus, W_min, H_lower, H_upper):
            """
            Returns the reduced wave energy in the given sublayer.
            """
            if m2_sl > 0:
                constant_part = (N ** 2) * (W_plus * np.conj(W_plus) + W_min * np.conj(W_min)) * (H_upper - H_lower)
                exp_part = - ((N ** 2) / 2) * 1j * (W_plus * np.conj(W_min) * (
                            np.exp(2 * 1j * sqrt(m2_sl) * H_upper) - np.exp(2 * 1j * sqrt(m2_sl) * H_lower)) / (
                                                                2 * sqrt(m2_sl))
                                                    - np.conj(W_plus) * W_min * (
                                                                np.exp(-2 * 1j * sqrt(m2_sl) * H_upper) - np.exp(
                                                            -2 * 1j * sqrt(m2_sl) * H_lower)) / (2 * sqrt(m2_sl)))
                return (constant_part + exp_part).real
            else:
                constant_part = ((N ** 2) / 2) * (W_plus * np.conj(W_min) + np.conj(W_plus) * W_min) * (
                            H_upper - H_lower)
                exp_part = - (N ** 2) * 1j * (W_plus * np.conj(W_plus) * (
                            np.exp(2.0 * 1j * sqrt(m2_sl) * H_upper) - np.exp(2.0 * 1j * sqrt(m2_sl) * H_lower)) / (
                                                          2 * sqrt(m2_sl))
                                              - np.conj(W_min) * W_min * (
                                                          np.exp(-2.0 * 1j * sqrt(m2_sl) * H_upper) - np.exp(
                                                      -2.0 * 1j * sqrt(m2_sl) * H_lower)) / (2 * sqrt(m2_sl)))
                # return exp_part.real
                return (constant_part + exp_part).real

        E_uni = np.zeros(shape, dtype=np.complex128)
        for indexk, k in enumerate(ks):
            for indexl, l in enumerate(ls):
                sigma3 = abl.U3*k+abl.V3*l
                #Non-hydrostatic solution
                if not sigma3==0:
                    if sigma3**2>abl.N**2:
                        m = 1j*np.sqrt((k**2+l**2)*np.abs(abl.N**2/sigma3**2-1))
                        E_uni[indexk, indexl] = energyLayer((m ** 2).real, abl.N, 1j * sigma3, 0, abl.H, abl.h_strat)
                        if E_uni[indexk, indexl] < 0:
                            print(E_uni[indexk, indexl])
                    else:
                        m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(abl.N**2/sigma3**2-1))
                        if np.sign(sigma3) > 0:
                            E_uni[indexk, indexl] = energyLayer((m ** 2).real, abl.N, 1j * sigma3, 0, abl.H, abl.h_strat)
                        else:
                            E_uni[indexk, indexl] = energyLayer((m ** 2).real, abl.N, 0, 1j * sigma3, abl.H, abl.h_strat)
                        if E_uni[indexk, indexl] < 0:
                            print(E_uni[indexk, indexl])
                else:
                    E_uni[indexk, indexl] = 1
        E = np.zeros(shape, dtype=np.complex128)

        # Get piecewise method inputs
        N_param, u_param, v_param, d2u_param, d2v_param, h_int, u_int, v_int, du_int, dv_int = self.pcm_input_setup(abl)

        for indexk, k in enumerate(ks):
            for indexl, l in enumerate(ls):
                sigma3 = u_int[0] * k + v_int[0] * l
                if not sigma3 == 0.:
                    amps = nus.amplitudes_numba(N_param, u_param, v_param, d2u_param, d2v_param,
                                                h_int, u_int, v_int, du_int, dv_int,
                                                k, l)
                    amps *= 1j*sigma3
                    critical_level = int((amps.size / 2) - 1)
                    m2 = nus.m_squared(N_param[:critical_level],
                                       u_param[:critical_level], v_param[:critical_level],
                                       d2u_param[:critical_level], d2v_param[:critical_level],
                                       k, l)
                    # Check resonance (through energy)
                    for sl in range(0, critical_level):
                        E[indexk, indexl] += energyLayer(m2[sl], N_param[sl], amps[2*sl], amps[2*sl+1], h_int[sl], h_int[sl+1])
                        if E[indexk, indexl] < 0:
                            print(E[indexk, indexl])
                else:
                    E[indexk, indexl] = 1
        A = np.divide(E, E_uni)
        return A

    def get_waves(self, abl, kv, lv):
        """
        Return the wave information for a given k and l vector.
        The returned amplitudes are those of the up- and down-going components of the vertical velocity perturbation.

        Parameters
        ----------
        kv  Wave-numbers in the x-direction
        lv  Wave-numbers in the y-direction

        Returns
        -------
        m_squared   Square of the vertical wave-numbers
        amplitudes  Amplitudes of the up- and down-going waves (amplitudes of vertical velocity)
        """
        if not self.dynamic:
            raise Exception("Hydrostatic upper atmosphere not implemented!")
        if self.rotating:
            raise Exception("Rotating upper atmosphere not implemented!")
        # Get piecewise method inputs
        N, u, v, d2u, d2v, h, u_int, v_int, du, dv = self.pcm_input_setup(abl)
        # Wave calculation
        amplitudes, m_squared = self.wave_calculation(kv, lv, N, u, v, d2u, d2v, h, u_int, v_int, du, dv)
        return h, m_squared, amplitudes

    def layer_setup(self, abl):
        """Set up the layer definitions used by the piecewise method"""
        # Get relevant profile range
        h_min = max(abl.H, abl.inv_top)
        h_max = abl.h_strat
        # Maximum altitude
        uci = abl.us[abl.zs < abl.h_strat]
        vci = abl.vs[abl.zs < abl.h_strat]
        Nci = abl.Ns[abl.zs < abl.h_strat]
        zci = abl.zs[abl.zs < abl.h_strat]
        # Minimum altitude
        uci = uci[zci > h_min]
        vci = vci[zci > h_min]
        Nci = Nci[zci > h_min]
        zci = zci[zci > h_min]
        # Set up interfaces and heights
        heights = zci
        dz = np.diff(heights)
        dz = np.append(2 * (h_min - heights[0]), -dz)
        h_interfaces = heights + dz / 2.
        # Interpolating data and setting up functions #
        # Brunt-Vaisala frequency (N)
        N_spline = interpolate.interp1d(zci, Nci, kind='nearest', fill_value=(Nci[0], abl.Ninf), bounds_error=False)
        # Velocity splines for easy derivatives
        U_spline = UnivariateSpline(zci, uci - abl.Uinf, k=self.order, s=0, ext=1)
        V_spline = UnivariateSpline(zci, vci - abl.Vinf, k=self.order, s=0, ext=1)
        dU_spline = lambda x: 0. * x
        d2U_spline = lambda x: 0. * x
        dV_spline = lambda x: 0. * x
        d2V_spline = lambda x: 0. * x
        if self.order > 0:
            dU_spline = U_spline.derivative()
            dV_spline = V_spline.derivative()
        if self.order > 1:
            d2U_spline = dU_spline.derivative()
            d2V_spline = dV_spline.derivative()
        # Use interp1d to ensure state in ABL and stratosphere is correct
        U_spline = interpolate.interp1d(zci, uci, kind=self.order, fill_value=(uci[0], abl.Uinf), bounds_error=False)
        V_spline = interpolate.interp1d(zci, vci, kind=self.order, fill_value=(vci[0], abl.Vinf), bounds_error=False)
        # User-defined number of layers
        if self.n_layers > 0:
            h_interfaces = np.linspace(h_min, h_max, self.n_layers + 1)
            dz = np.diff(h_interfaces)
            dz = np.append(dz, dz[-1])
            heights = h_interfaces + dz / 2.0
            def height_average(spline, new_heights, edges, fill_value):
                new_values = spline(new_heights)
                new_values = np.append(np.append(new_values[0], new_values), new_values[-1])
                interpol_heights = np.append(np.append(edges[0], new_heights), edges[-1])
                return interpolate.interp1d(interpol_heights, new_values, kind='nearest', fill_value=fill_value, bounds_error=False)
            U_spline = height_average(U_spline, heights, [h_interfaces[0], h_interfaces[-1]], (uci[0], abl.Uinf))
            dU_spline = lambda z : 0. * z
            d2U_spline = lambda z : 0. * z
            V_spline = height_average(V_spline, heights, [h_interfaces[0], h_interfaces[-1]], (vci[0], abl.Vinf))
            dV_spline = lambda z : 0. * z
            d2V_spline = lambda z : 0. * z
            N_spline = height_average(N_spline, heights, [h_interfaces[0], h_interfaces[-1]], (Nci[0], abl.Ninf))
        return heights, h_interfaces, N_spline, U_spline, V_spline, dU_spline, dV_spline, d2U_spline, d2V_spline

    def pcm_input_setup(self, abl):
        """Set up the inputs for the piece-wise constant method"""
        # Get layer definitions and splines
        heights, h_interfaces, N, U, V, dU, dV, d2U, d2V = self.layer_setup(abl)
        # Set up piecewise method inputs #
        # Slight shift compared to interface, to evaluate conditions at both sides
        epsilon = (h_interfaces[1] - h_interfaces[0]) / 100.
        h_lim = np.append(np.append(h_interfaces[0], h_interfaces[1:] - epsilon), h_interfaces[1:] + epsilon)
        h_lim = np.sort(h_lim)
        # Sublayer parameters
        u_param = U(heights)
        v_param = V(heights)
        d2u_param = d2U(heights)
        d2v_param = d2V(heights)
        N_param = N(heights)
        # Sublayer interface parameters
        u_int = U(h_lim)
        v_int = V(h_lim)
        du_int = dU(h_lim)
        dv_int = dV(h_lim)
        return N_param, u_param, v_param, d2u_param, d2v_param, h_interfaces, u_int, v_int, du_int, dv_int

    def FA_plot(self, abl):
        """Plot the upper atmospheric profiles"""
        # Relevant domain
        h_min = max(abl.H, abl.inv_top)
        h_plot = 15.e3
        # Vertical grid
        z_plot = np.linspace(abl.H, h_plot, 15000)
        # Get ABL data
        domain = np.logical_and(h_min < abl.zs, abl.zs < h_plot)
        altitudes = abl.zs[domain]
        N_values = abl.Ns[domain]
        u_values = abl.us[domain]
        v_values = abl.vs[domain]
        # Get layer definitions and splines
        heights, h_interfaces, N, U, V, dU, dV, d2U, d2V = self.layer_setup(abl)
        # Figure
        fig, axarr = plt.subplots(1, 3, figsize=(14., 5.5))
        # Brunt-Vaisala frequency
        axarr[0].plot(N(z_plot), z_plot/1.e3, '-k')
        axarr[0].plot(N_values, altitudes/1.e3, 'x')
        axarr[0].set_ylabel('$z$ [km]')
        axarr[0].set_xlabel('$N_g$ [s$^{-1}$]')
        axarr[0].set_ylim([h_min/1.e3, h_plot/1.e3])
        # Velocity x-direction
        axarr[1].plot(U(z_plot), z_plot/1.e3, '-k')
        axarr[1].plot(u_values, altitudes/1.e3, 'x')
        axarr[1].set_xlabel('$U$ [m/s]')
        axarr[1].set_ylim([h_min/1.e3, h_plot/1.e3])
        # Velocity y-direction
        axarr[2].plot(V(z_plot), z_plot/1.e3, '-k')
        axarr[2].plot(v_values, altitudes/1.e3, 'x')
        axarr[2].set_xlabel('$V$ [m/s]')
        axarr[2].set_ylim([h_min/1.e3, h_plot/1.e3])
        return fig, axarr

    @staticmethod
    @njit(parallel=False)
    def wave_calculation(ks, ls, N, u, v, d2u, d2v, h, u_int, v_int, du, dv):
        # Grid shape
        Nx = len(ks)
        Ny = len(ls)
        # Parameter readout
        # Output setup
        amplitudes = np.zeros((Nx, Ny, len(N) * 2), dtype=np.complex128)
        m_squared = np.zeros((Nx, Ny, len(N)), dtype=np.complex128)
        # Loop over wavenumbers
        for indexk in numba.prange(ks.size):
            for indexl in numba.prange(ls.size):
                sigma = u_int[0] * ks[indexk] + v_int[0] * ls[indexl]
                if not sigma == 0:
                    amps = nus.amplitudes_numba(N, u, v, d2u, d2v, h, u_int, v_int, du, dv, ks[indexk], ls[indexl])
                    amplitudes[indexk, indexl, :len(amps)] = amps
                    m_squared[indexk, indexl] = nus.m_squared(N, u, v, d2u, d2v,
                                                              ks[indexk], ls[indexl])
        return amplitudes, m_squared


    @staticmethod
    @njit(parallel=False)
    def evaluate_Phi(shape, ks, ls, gprime, N, u, v, d2u, d2v, h, u_int, v_int, du, dv, dynamic=True):
        '''
        Compute complex stratification coefficient Phi for height-dependent atmospheres

        Parameters
        ----------
        shape
            shape of the grid
        ks: array-like
            k-vector
        ls: array-like
            l-vector
        gprime: float
            inversion layer strength
        N: array
            The Brunt-Vaisala frequency N_g, evaluated in each layer.
            Used to determine the vertical wavenumber in each layer.
        u: array
            The wind speed in the x-direction, evaluated in each layer.
            Used to determine the vertical wavenumber in each layer.
        v: array
            The wind speed in the y-direction, evaluated in each layer.
            Used to determine the vertical wavenumber in each layer.
        d2u: array
            The second derivative with respect to height of the wind speed in the x-direction, evaluated in each layer.
            Used to determine the vertical wavenumber in each layer.
        d2vN: array
            The second derivative with respect to height of the wind speed in the y-direction, evaluated in each layer.
            Used to determine the vertical wavenumber in each layer.
        h: array
            The altitudes of the interfaces (in order).
            Used to set up the sublayer interface matching conditions.
        u_int: array
            The wind speed in the x-direction, evaluated at the layer interfaces.
            Used to set up the sublayer interface matching conditions.
        v_int: array
            The wind speed in the y-direction, evaluated at the layer interfaces.
            Used to set up the sublayer interface matching conditions.
        du: array
            The derivative with respect to height of the wind speed in the x-direction,
            evaluated at the layer interfaces.
            Used to set up the sublayer interface matching conditions.
        dv: array
            The derivative with respect to height of the wind speed in the y-direction,
            evaluated at the layer interfaces.
            Used to set up the sublayer interface matching conditions.
        dynamic: bool
            Whether dynamic effects are taken into account

        Returns
        -------
        PHI: 2d numpy array
            complex stratification coefficient
        '''
        PHI = gprime*np.ones(shape, dtype=np.complex128)
        for indexk in numba.prange(ks.size):
            for indexl in numba.prange(ls.size):
                sigma3 = u_int[0]*ks[indexk] + v_int[0]*ls[indexl]
                if not sigma3 == 0.:
                    PHI[indexk, indexl] += nus.phiConstant(N, u, v, d2u, d2v, h, u_int, v_int, du, dv,
                                                           ks[indexk], ls[indexl],
                                                           dynamic)
        return PHI

    @property
    def n_layers(self):
        return self.__n_layers

    @property
    def order(self):
        return self.__order
