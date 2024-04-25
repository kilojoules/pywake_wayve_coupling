#!/usr/bin/env python

'''
Module containing common setup routines for ABL objects.
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "October 11, 2023"

import numpy as np
from scipy import interpolate
import mpmath
import warnings
import matplotlib.pyplot as plt

from wayve.abl.abl import ABL
from wayve.abl import line_optimization, ci_methods
from wayve.abl.abl_tools import Cg_cubic, alpha_cubic


def analytic_cubic(dth, fc, N, G, geo_angle, kappa, utau, h, z0, H1, TI, Nz):
    '''
    Method to specify the atmospheric state based on analytic profiles with a cubic eddy viscosity profile.

    The analytic profiles have been derived by Nieuwstadt 1983 [1].

    Parameters
    ----------
    dth: float
        Inversion strength
    fc: float
        Coriolis parameter
    N: float
        Brunt Vaisala frequency
    G: float
        Geostrophic wind speed
    geo_angle: float
        Geostrophic wind direction
    kappa: float
        Von Karman constant
    utau: float
        Friction velocity
    h: float
        Boundary-layer height
    z0: float
        Surface roughness
    TI: float
        Turbulence intensity
    H1: float
        Height of the wind-farm layer
    Nz: int
        Number of vertical gridpoints in the ABL

    References
    ----------
    [1] Nieuwstadt, F. T. M. (1983). On the solution of the stationary, baroclinic Ekman-layer equations with a finite
        boundary-layer height. Boundary-Layer Meteorology, 26(4), 377–390. https://doi.org/10.1007/BF00119534
    '''
    # Upper atmosphere
    U3 = G * np.cos(geo_angle)
    V3 = G * np.sin(geo_angle)

    # Vertical grid
    zs = np.linspace(z0, h, Nz)

    # Nieuwstadt solution #
    C = h * fc / kappa / utau
    alpha = 0.5 + 0.5 * np.sqrt(1 + 4j * C)
    sigma_s = np.zeros(len(zs), dtype=np.complex128)
    wd_s = np.zeros(len(zs), dtype=np.complex128)
    with np.errstate(invalid='ignore'):  # z>=h will result in Nan. This is set to 0 below.
        for k in range(len(zs)):
            sigma_s[k] = alpha * (mpmath.gamma(alpha)) ** 2 / mpmath.gamma(2 * alpha) \
                         * np.power(1. - zs[k] / h, alpha) \
                         * mpmath.hyp2f1(alpha - 1, alpha, 2 * alpha, 1 - zs[k] / h)
            wd_s[k] = (1j * alpha ** 2 * (mpmath.gamma(alpha)) ** 2) / (kappa * C * mpmath.gamma(2 * alpha)) * (
                    1 - zs[k] / h) ** (alpha - 1) * mpmath.hyp2f1(alpha + 1, alpha - 1, 2 * alpha,
                                                                  1 - zs[k] / h)
    # Set Nan to 0
    sigma_s[np.isnan(sigma_s)] = np.complex128(0.)
    wd_s[np.isnan(wd_s)] = np.complex128(0.)

    # Velocity arrays
    us = U3 + np.real(wd_s) * utau
    vs = V3 + np.imag(wd_s) * utau

    # Momentum flux arrays
    tauxs = np.real(sigma_s) * utau ** 2
    tauys = np.imag(sigma_s) * utau ** 2
    nus = kappa * utau * np.multiply(zs, (1 - zs / h) ** 2, out=np.zeros_like(zs), where=(zs <= h))

    # Capping inversion reduced gravity
    gprime = 9.81 * dth / 288.15

    # Potential temperature array
    ths = 0. * zs + 288.15  # No stratification within ABL

    # Upper layer thickness
    H2 = h - H1

    # Set up ABL object
    abl = ABL(zs, us, vs, ths, tauxs, tauys,
              H1, H2,
              gprime, N, U3, V3,
              fc,
              nus=nus, TI=TI, z0=z0, ust=utau)

    return abl


def AM2019(sub, Nz=1000):
    '''
    Set up an ABL object corresponding to the examples used by Allaerts and Meyers (2019) [1].

    Parameters
    ----------
    sub: bool
        Whether the sub- or supercritical case is used
    Nz: int (optional)
        Number of vertical gridpoints in the ABL (default: 1000)
        Allaerts and Meyers (2019) used Nz=100.

    References
    ----------
    [1] Allaerts, D., & Meyers, J. (2019). Sensitivity and feedback of wind-farm-induced gravity waves.
        Journal of Fluid Mechanics, 862, 990–1028. https://doi.org/10.1017/jfm.2018.969
    '''
    # Universal constants #
    kappa = 0.41
    A = 500
    # Non-dimensional inputs
    hstar = 0.15  # Non-dimensional boundary layer height [-]
    z0_h = 1.e-4  # Non-dimensional surface roughness length [-]
    Nfc = 58  # Brunt-Vaisala frequency to Coriolis parameter
    if sub:  # Froude number 0.9 or 1.1
        inv_par = 1.04  # Inversion parameter
    else:
        inv_par = 0.69  # Inversion parameter
    TI = 0.12  # Turbulence intensity
    # Nieuwstadt relations
    Cg = Cg_cubic(hstar, z0_h, kappa)  # Geostrophic drag Cg = utau/G
    alpha = alpha_cubic(hstar, z0_h, kappa)  # Geostrophic wind angle
    # Scaling parameters
    fc = 1.0e-4  # Coriolis parameter [1/s]
    H = 1.e3  # Boundary layer height
    H1 = 240.  # Wind-farm layer height
    # Dimensional parameters
    utau = H * fc / hstar
    N = Nfc * fc
    z0 = H * z0_h
    G = utau / Cg
    dth = 288.15 * inv_par * A * utau ** 2 / (H * 9.81)
    # Set up ABL with Nieuwstadt profile
    abl = analytic_cubic(dth, fc, N, G, alpha, kappa, utau, H, z0, H1, TI, Nz)
    # Rotate abl so that wind is aligned in the wind-farm layer
    abl.rotate(abl.WD1*np.pi/180.0)
    return abl


def mesoscale_based(zs, us, vs, ths,
                    ust, blh, l_mo, lat,
                    H1,
                    z0=None, TI=None,
                    rho=1.225,
                    dh_max=300., Gmode="avg", serz=True,
                    plot_fits=False):
    '''
    Method to specify the atmospheric state based on data typically outputted by mesoscale models such as COSMO-CCLM, or
    re-analysis data such as ERA5.

    This method processes the given variables in three key ways, in order to make it useable for WAYVE:
        - It sets up a basic profile for the vertical momentum fluxes and eddy viscosities, based on the friction
          velocity, turbulent boundary layer height, and Monin-Obhukov length.
        - It determines the height and strength of the capping inversion and the free lapse rate, based on the potential
          temperature profile, using the (surface-extended) Rampanelli and Zardi model.
        - It determines the height of the tropopause and the velocity and lapse rate in the stratosphere, based on the
          potential temperature profile.
    Additionally, various parameters such as the surface roughness height and turbulence intensity are estimated, if
    they are not specified by the user.

    Parameters
    ----------
    zs: 1d numpy array
        Altitudes at which velocity and temperature profiles are defined
    us,vs: 1d numpy array
        Velocity profiles in x and y directions
    ths: 1d numpy array
        Potential temperature profile
    ust: float
        Friction velocity
    blh: float
        Height of the boundary layer, as based on the turbulent momentum fluxes
    l_mo: float
        Monin-Obhukov length
    lat: float
        Latitude coordinate (in degrees), used to determine the Coriolis parameter
    H1: float
        Height of the wind-farm layer
    z0 (optional): float
        Surface roughness height (default: calculated from profiles)
    TI (optional): float
        Turbulence intensity (default: calculated from profiles)
    rho (optional): float
        Air density (default: 1.225kg/m3)
    dh_max (optional): float
        Maximum depth of the inversion layer used in the inversion curve fitting procedure (default: 300m)
    Gmode (optional): str
        Method to define free atmosphere velocity
        "h1": take velocity at h1 (inversion center)
        "h2": take velocity at h2 (inversion top)  (h2=h1+Deltah/2)
        "top": take velocity at 5000 m
        "avg": average velocity profile between h1 and 5000 m
        "trop": average velocity profile between h1 and the tropopauze. If used, N is based on the tropopause fit.
        default: "avg"
    serz (optional): bool
        Whether the surface-extended version of the RZ model is used for the CI fitting (default: True)
    plot_fits (optional): bool
        Whether the fitting of the CNBL structure and the tropopause is plotted (default: False)
    '''
    # Constants
    gravity = 9.80665  # [m s-2]
    kappa = 0.41  # Von Karman constant
    omega = 7.2921159e-5  # angular speed of the Earth [rad/s]

    # Total velocity
    Ms = np.sqrt(us ** 2 + vs ** 2)

    # Stable or unstable atmosphere
    stable = l_mo > 100

    # Estimate inversion parameters with RZ fit #
    # Relevant part of the vertical profiles
    max_z_fit = 5.e3
    z_ci = zs[zs <= max_z_fit]
    th_ci = ths[zs <= max_z_fit]
    # Surface-Extended RZ or regular RZ
    if serz:
        # Stable or unstable profile determines the initial guess for the CI height
        if stable:
            l_p0 = 1.e3
        else:
            l_p0 = blh
        # Initial estimate for MBL temperature in fit
        th_mbl = np.interp(l_p0, z_ci, th_ci)
        # Fitting procedure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ci_estimate = ci_methods.SERZ_fit(z_ci, th_ci,
                                              p0=[0.9, .1, 0., th_mbl, l_p0, 100, .05],
                                              initialGuess='RZ',
                                              dh_max=dh_max)
    else:
        # Stable or unstable profile
        if stable:
            # Ignore temperature decrease inside SBL
            # (we are trying to identify the mixing layer that preceded this SBL)
            # We want to capture the mixed layer (or residual layer since we are in
            # SBL) that preceded the SBL. Therefore, we take the potential temperature
            # at the top of the ABL where we have the mixed layer and we extrapolate
            # till the bottom. We use this constant value in the ABL.
            # p0 are the initial guess for [a,b,thm,l,dh] used in Ramp&Zar model
            th_ci[z_ci < blh] = np.interp(blh, z_ci, th_ci)
            l_p0 = 1.e3
        else:
            # Ignore temperature increase in CBL surface layer, therefore we take
            # the lowest value of potential temperature. We are able to capture the
            # mixed layer in this way, where the temperature is constant and equal
            # to the lowest theta. We use this constant value in the ABL.
            # p0 are the initial guess for [a,b,thm,l,dh] used in Ramp&Zar model
            th_ci[0:np.argmin(th_ci)] = np.min(th_ci)
            l_p0 = blh
        # Initial estimate for MBL temperature in fit
        th_mbl = np.interp(l_p0, z_ci, th_ci)
        # RZ fit
        ci_estimate = ci_methods.RZfit(z_ci, th_ci,
                                       p0=[0.9, 0.1, th_mbl, l_p0, 100.0],
                                       dh_max=dh_max)
    # Plot fitted potential temperature profile
    if plot_fits:
        fig, ax = plt.subplots()
        ax.plot(ths[zs <= max_z_fit], z_ci / 1.e3, 'b', label="Data")
        ax.plot(ci_estimate['thfit'], z_ci / 1.e3, '--k', label="RZ fit", zorder=-1)
        ax.set_xlim([285., 312.])
        ax.set_ylim([0., 4.])
        ax.set_ylabel('$z$ [km]')
        ax.set_xlabel('$\\theta$ [K]')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # CI altitudes
    inv_bottom = ci_estimate['h0']
    H = ci_estimate['h1']
    inv_top = ci_estimate['h2']
    # Determine reference potential temperature
    th0 = np.interp(H, zs, ths)
    # Inversion strength
    if ci_estimate['a'] <= 0.2 or ci_estimate['a'] <= 2 * ci_estimate['b']:
        # No inversion strength in the following cases:
        # a<=0.2: encroachment (No inversion layer, so the entire profile is given by g and a=0
        #           (considered a,0.2 as in paper))
        # a<=2*b: inversion lapse rate is equal to or smaller than free lapse rate
        gprime = 0.
    else:
        gprime = gravity * ci_estimate['dth'] / th0
    if gprime == 0.:
        raise RuntimeWarning("No CI present!")
    # Brunt-Vaisala frequency
    N = np.sqrt(gravity * ci_estimate['gamma'] / th0)
    # Upper layer thickness
    H2 = H - H1
    if H2 < 10.:
        raise RuntimeWarning("CI too low!")

    # Upper atmosphere variable determination #
    # Approximate the troposphere-stratosphere structure of the temperature profile using two lines
    nLines = 2
    # Select relevant region of the temperature profile
    fa_selection = np.logical_and(H < zs, zs < 15000.)
    altitudes = zs[fa_selection]
    theta_values = ths[fa_selection]
    u_values = us[fa_selection]
    v_values = vs[fa_selection]
    # Set up an array of dimensions n x 2 with profile1[:,0] the temperature and profile1[:,1] the heights
    profile = np.stack((theta_values, altitudes), axis=1)
    profiles = [profile]
    # Perform 2-line fit
    results = line_optimization.smart_slsqp_optimization(nLines, profiles, verbose=0)
    # Read out results
    theta_trop = results[0][0]
    a1 = results[0][1]
    dthetadz_trop = a1
    h_strat = results[0][2]
    a2 = results[0][3]
    theta_strat = theta_trop + a1 * h_strat
    # Stratosphere conditions
    u_strat = np.mean(u_values[altitudes > h_strat])
    v_strat = np.mean(v_values[altitudes > h_strat])
    dthetadz_strat = a1 + a2
    # Brunt-Vaisala frequency in the troposphere and stratosphere
    N_trop = np.sqrt(gravity * dthetadz_trop / theta_trop)
    N_strat = np.sqrt(gravity * dthetadz_strat / theta_strat)
    # Plot tropopause line fit
    if plot_fits:
        # Set up figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # theta
        ax1.plot(profile[:, 0], profile[:, 1] / 1.e3)
        ax1.plot(line_optimization.theta_approximation(nLines, profile[:, 1], results[0]), profile[:, 1] / 1.e3,
                 '--k')
        ax1.set_ylabel('$z$ [km]')
        ax1.set_xlabel('$\\theta$ [K]')
        # u
        ax2.plot(u_values, altitudes / 1.e3)
        ax2.plot([u_strat, u_strat], [h_strat / 1.e3, 15000. / 1.e3], '--k')
        ax2.set_xlabel('$u$ [m/s]')
        # v
        ax3.plot(v_values, altitudes / 1.e3)
        ax3.plot([v_strat, v_strat], [h_strat / 1.e3, 15000. / 1.e3], '--k')
        ax3.set_xlabel('$v$ [m/s]')
        # Show plot
        plt.tight_layout()
        plt.show()

    # Compute FA quantities #
    if Gmode == 'h1':
        U3 = np.interp(ci_estimate['h1'], zs, us)
        V3 = np.interp(ci_estimate['h1'], zs, vs)
    elif Gmode == 'h2':
        U3 = np.interp(ci_estimate['h2'], zs, us)
        V3 = np.interp(ci_estimate['h2'], zs, vs)
    elif Gmode == 'top':
        U3 = np.interp(5.e3, zs, us)
        V3 = np.interp(5.e3, zs, vs)
    elif Gmode == 'avg':
        z = np.linspace(H, 5000., 1000)
        U3 = np.trapz(np.interp(z, zs, us), z) / (z[-1] - z[0])
        V3 = np.trapz(np.interp(z, zs, vs), z) / (z[-1] - z[0])
    elif Gmode == 'trop':
        z = np.linspace(H, h_strat, 1000)
        U3 = np.trapz(np.interp(z, zs, us), z) / (h_strat - H)
        V3 = np.trapz(np.interp(z, zs, vs), z) / (h_strat - H)
        N = N_trop  # Use N as determined by the tropopauze fit
    else:
        raise Exception("Gmode for ERA5-based ABL setup unknown")

    # Profiles for momentum fluxes and eddy viscosity #
    # Flux profile
    tau = np.zeros(zs.shape)
    nus = np.zeros(zs.shape)
    if stable:  # stable - turbulence up to the given boundary layer height
        tau[zs <= blh] = ust ** 2 * (1 - zs[zs <= blh] / blh) ** 1.5
        nus[zs <= blh] = kappa * ust * zs[zs <= blh] * (1 - zs[zs <= blh] / blh) ** 2
    else:  # unstable - turbulence up to the capping inversion
        tau[zs <= H] = ust ** 2 * (1 - zs[zs <= H] / H)
        nus[zs <= H] = kappa * ust * zs[zs <= H] * (1 - zs[zs <= H] / H) ** 2
    # Angle of the momentum flux
    tau_angle = np.arctan2(V3, U3)      # Assume momentum flux is aligned with the geostrophic wind
    # tau_angle = np.arctan2(vs, us)    # Assume eddy viscosity behavior
    # X and Y momentum flux components
    tauxs = np.multiply(np.cos(tau_angle), tau)
    tauys = np.multiply(np.sin(tau_angle), tau)

    # Various parameters #
    # Coriolis parameter
    phi = np.radians(lat)
    fc = 2 * omega * np.sin(phi)
    # Turbulent intensity
    if TI is None:
        # TKE profile
        if stable:  # stable
            # From Nieuwstadt (1984): q/sqrt(tau) = 3
            tke = 4.5 * tau
        else:   # unstable
            # From Stull (1988): q^2/tau = 8.5+2.5
            tke = 5.5 * tau
        # TI profile
        TIs = np.sqrt(2. / 3. * tke) / Ms
        # Interpolate TI in the middle of the lower layer
        TI = np.interp(H1/2., zs, TIs)
    # Surface roughness
    if z0 is None:
        z0 = zs[0] / np.exp(kappa * Ms[0] / ust)    # Ignoring stability effects

    # Set up ABL object #
    abl = ABL(zs, us, vs, ths, tauxs, tauys,
              H1, H2,
              gprime, N, U3, V3,
              fc,
              nus=nus, rho=rho, TI=TI, z0=z0, ust=ust,
              inv_bottom=inv_bottom, inv_top=inv_top, h_strat=h_strat, Uinf=u_strat, Vinf=v_strat, Ninf=N_strat)

    return abl
