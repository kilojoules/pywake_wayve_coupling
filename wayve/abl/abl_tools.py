#!/usr/bin/env python

'''
Some additional functions for ABL setup calculations.
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "June 23, 2023"

import mpmath
import numpy as np


def height_average(u, z, h1, h2):
    '''Integrate a given profile from h1 to h2,
    with z the location of the grid cell centers'''
    # Select array parts that are within the boundaries
    selection = np.logical_and(h1 <= z, z <= h2)
    u_select = u[..., selection]
    z_select = z[selection]
    # Linear interpolation to layer boundaries #
    # Lower boundary correction
    if h1 > z[0]:   # No correction if h1<z[0], as linear interpolation does not hold close to the ground
        u_h1 = np.interp(h1, z, u)
        z_select = np.insert(z_select, 0, h1)
        u_select = np.insert(u_select, 0, u_h1)
    # Upper boundary correction
    u_h2 = np.interp(h2, z, u)
    z_select = np.append(z_select, h2)
    u_select = np.append(u_select, u_h2)
    # Height average array
    integral = np.trapz(u_select, x=z_select, axis=-1) / (z_select[-1] - z_select[0])
    return integral


def Cg_cubic(hstar, z0_h, kappa):
    '''
    Geostrophic drag Cg = utau/G as a function of hstar = h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    z0_h: float
        Non-dimensional surface roughness length
        z0_h = z0/h
    kappa: float
        Von Karman constant

    Returns
    -------
    Cg: float
        Geostrophic drag
        Cg = utau/G
    '''
    F1 = F1_cubic(hstar,kappa)
    F2 = F2_cubic(hstar,kappa)
    Cg = (F2**2+(1./kappa*np.log(1./(hstar*z0_h))-F1)**2)**(-1/2)
    return Cg


def alpha_cubic(hstar, z0_h, kappa):
    '''
    Geostrophic wind direction alpha as a function of hstar = h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    z0_h: float
        Non-dimensional surface roughness length
        z0_h = z0/h
    kappa: float
        Von Karman constant

    Returns
    -------
    alpha: float
        Geostrophic wind direction
    '''
    Cg = Cg_cubic(hstar,z0_h,kappa)
    F2 = F2_cubic(hstar,kappa)
    alpha = np.arcsin(-Cg*F2)
    return alpha


def F1_cubic(hstar, kappa):
    '''
    F1 function corresponding to a cubic viscosity profile (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    kappa: float
        Von Karman constant

    Returns
    -------
    F1: float
        Value of F1 function
    '''
    C = hstar/kappa
    alpha = 0.5+0.5*np.sqrt(1+4j*C)
    F1 = np.zeros((1),dtype=np.float64)
    F1[0] = 1./kappa*(-np.log(hstar)+
                      mpmath.re( mpmath.digamma(alpha+1)+
                                 mpmath.digamma(alpha-1)-
                                 2*mpmath.digamma(1.0) ) )
    return F1.item()


def F2_cubic(hstar, kappa):
    '''
    F2 function corresponding to a cubic viscosity profile (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    kappa: float
        Von Karman constant

    Returns
    -------
    F2: float
        Value of F2 function
    '''
    C = hstar/kappa
    alpha = 0.5+0.5*np.sqrt(1+4j*C)
    F2 = np.zeros((1),dtype=np.float64)
    F2[0] = 1./kappa*(mpmath.im( mpmath.digamma(alpha+1)+
                                 mpmath.digamma(alpha-1)-
                                 2*mpmath.digamma(1.0) ) )
    return F2.item()
