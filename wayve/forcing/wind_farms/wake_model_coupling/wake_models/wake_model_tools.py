#!/usr/bin/env python

'''
Some implementations shared by multiple wake models and merging methods
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "October 15, 2023"

import numba
import numpy as np
from numba import njit


@njit(parallel=False)
def evaluate_TI(Nt, e_str, e_span, order, x, y, D, Ct, TIinf, ka=0.3837, kb=0.003678):
    '''
    Turbulence intensity model of Niayifar and Porte-Agel (2016) in Numba syntax

    Turbulence intensity is assumed not to depend on the perturbation velocities so
    that, contrary to the Gaussian wake model itself, the turbulence intensity model
    can be calculated by ordering the turbines and then sweeping through the farm

    Parameters
    ----------
    Nt: float
        Number of wind turbines
    e_str: 1d numpy array
        Unit vector along the wind direction
    e_span: 1d numpy array
        Unit vector in cross wind direction
    order: 1d numpy array
        Wind Turbine index ordered according to wind direction
    x: 1d numpy array
        Wind Turbine x-coordinates
    y: 1d numpy array
        Wind Turbine y-coordinates
    D: 1d numpy array
        Wind Turbine diameters
    Ct: 1d numpy array
        Wind Turbine thrust coefficients
    TIinf: float
        Free stream turbulence intensity
    ka: float (optional)
        Wake model parameter (default 0.3837, from Niayifar and Porte-Agel (2016))
    kb: float (optional)
        Wake model parameter (default 0.003678, from Niayifar and Porte-Agel (2016))

    Returns
    -------
    TI: 1d numpy array
        Local turbulent intensity at the turbines
    '''
    TI = np.zeros(Nt)
    TI[0] = TIinf
    for i in numba.prange(1, Nt):
        TIadded = np.zeros(i)
        maxTIadded = 0
        # Compute added streamwise turbulent intensity induced by Turbine k
        # at Turbine i
        for k in numba.prange(0, i):
            kwake = ka * TI[k] + kb
            # Vector from Turbine K to point n on Turbine I
            KI = np.array([x[i] - x[k], y[i] - y[k]])
            # Streamwise distance between Turbine K and Turbine I along e_str
            # Positive if I is downstream of K
            # Negative if I is upstream of K
            delta_str = np.dot(KI, e_str)
            # Spanwise distance between Turbine K and Turbine I along e_span
            # Has a sign but wake model is axisymmetric
            delta_span = np.dot(KI, e_span)
            # Turbine k lies upstream of Turbine i
            if delta_str > 0:  # and delta_str<15*turbk.D:
                beta = 0.5 * (1 + np.sqrt(1 - Ct[k])) / np.sqrt(1 - Ct[k])
                eps = 0.2 * np.sqrt(beta)
                sigma = kwake * delta_str + D[k] * eps
                rwake = 2 * sigma
                rdisk = D[i] / 2.0
                # TI wake of k intersects with rotor i
                # the wake start at the edge of the Turbine rotor, so the width of the wake
                # is given by rdisk + the width of the wake, so rwake
                # If the lateral distance between Turbine is greater than rwake+rdisk,
                # then there is no overlap
                if rwake >= rdisk:
                    maximum = rwake
                    minimum = rdisk
                else:
                    maximum = rdisk
                    minimum = rwake
                if np.abs(delta_span) >= (rwake + rdisk):
                    Aw = 0.  # No overlap
                elif np.abs(delta_span) >= maximum:
                    xt = (delta_span ** 2 - rdisk ** 2 + rwake ** 2) / (2 * np.abs(delta_span))
                    Aw = (area_circle_segment(rwake, xt)
                          + area_circle_segment(rdisk, np.abs(delta_span) - xt))
                elif np.abs(delta_span) >= np.abs(rwake - rdisk):
                    R = maximum
                    r = minimum
                    xt = (delta_span ** 2 - r ** 2 + R ** 2) / (2 * np.abs(delta_span))
                    Aw = (area_circle_reflex(r, xt - np.abs(delta_span))
                          + area_circle_segment(R, xt))
                else:
                    Aw = np.pi * minimum ** 2  # Area of overlapping - The overlap cannot be smaller than rdisk

                induction = (1 - np.sqrt(1 - Ct[k])) / 2.
                Iadded = (0.73 * induction ** (0.8325)
                          * TIinf ** (0.0325)
                          * (delta_str / D[k]) ** (-0.32))
                TIadded[k] = Aw / (np.pi * rdisk ** 2) * Iadded
                if TIadded[k] > maxTIadded:
                    maxTIadded = TIadded[k]
        TI[i] = np.sqrt(TIinf ** 2 + (maxTIadded) ** 2)

    # Unsort turbines
    inv_order = np.argsort(order)
    return TI[inv_order]


@njit
def area_circle_segment(R, d):
    '''Area of a circle segment when the angle is less than 180°'''
    return R ** 2 * np.arccos(d / R) - d * np.sqrt(R ** 2 - d ** 2)


@njit
def area_circle_reflex(R, d):
    '''Area of a circle segment when the angle is greater than 180°'''
    alpha = np.arccos(d / R)
    return (np.pi - alpha) * R ** 2 + d * np.sqrt(R ** 2 - d ** 2)