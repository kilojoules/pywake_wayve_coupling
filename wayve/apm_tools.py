#!/usr/bin/env python

'''
Some additional functions
'''

import numpy as np
from numba import njit, NumbaPerformanceWarning
import numba

# Suppress warning about matrix-vector product performance, as it is not a major issue in the code
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "August 22, 2017"


@njit(parallel=False)
def evaluate_M(ks2, ls, U1, V1, U2, V2, H1, H2, nu1, nu2, fc, PHI, tau_jac, N2, y):
    '''
    Solve the atmospheric perturbation model preconditioner in Numba syntax

    Parameters
    ----------
    ks2,ls: numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    H1: float
        Height of the wind-farm layer
    H2: float
        Height of the upper layer
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    fc: float
        Coriolis parameter
    PHI: 2d numpy array
        Complex stratification coefficients
    tau_jac: numpy array
        Jacobians of the momentum flux terms
    N2: int
        Total number of grid points
    y: numpy array
        APM right-hand side

    Returns
    -------
    M: 2d numpy array
        Preconditioner
    '''
    # Re-shape APM state
    Y = y.reshape(6, N2)
    # Set up output
    M = np.zeros((6, N2), dtype=np.complex128)
    # Grid size
    Nx2 = ks2.size
    Ny = ls.size
    # Loop over wavenumbers
    for indexk in numba.prange(Nx2):
        for indexl in numba.prange(Ny):
            # Get index
            index = indexl + Ny*indexk
            # Set up preconditioner matrix
            P = P_per_WN(ks2[indexk], ls[indexl], U1, V1, U2, V2, H1, H2, nu1, nu2, fc, PHI[indexk, indexl], tau_jac)
            # Solve system
            M[:,index] = np.linalg.solve(P, Y[:, index])        # BEST - Use on SPC (set parallel=False)
            # M[:,index] = np.dot(np.linalg.inv(P),Y[:,index])  # BEST - Use on PC
    # Re-shape output
    M = M.reshape(6*N2)
    return M


@njit(parallel=False)
def evaluate_P(ks2, ls, U1, V1, U2, V2, H1, H2, nu1, nu2, fc, PHI, tau_jac, N2, x):
    '''
    Evaluate atmospheric perturbation model preconditioner in Numba syntax.

    In contrast to the evaluate_M function, which solves the system Px=y, this function evaluates Px for a given x.

    Parameters
    ----------
    ks2,ls: numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    H1: float
        Height of the wind-farm layer
    H2: float
        Height of the upper layer
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    fc: float
        Coriolis parameter
    PHI: 2d numpy array
        Complex stratification coefficients
    tau_jac: numpy array
        Jacobians of the momentum flux terms
    N2: int
        Total number of grid points
    x: numpy array
        APM state

    Returns
    -------
    P: 2d numpy array
        Preconditioner matrix product
    '''
    # Re-shape APM state
    X = x.reshape(6, N2)
    # Set up output
    P = np.zeros((6, N2), dtype=np.complex128)
    # Grid size
    Nx2 = ks2.size
    Ny = ls.size
    # Loop over wavenumbers
    for indexk in numba.prange(Nx2):
        for indexl in numba.prange(Ny):
            # Get index
            index = indexl + Ny*indexk
            # Set up preconditioner matrix
            P_kl = P_per_WN(ks2[indexk], ls[indexl], U1, V1, U2, V2, H1, H2, nu1, nu2, fc, PHI[indexk, indexl], tau_jac)
            # Calculate output
            P[:,index] = np.dot(P_kl, X[:, index])
    # Re-shape output
    P = P.reshape(6*N2)
    return P


@njit(parallel=False)   # Setting this to True slows the code down with a factor of 3
def P_per_WN(k, l, U1, V1, U2, V2, H1, H2, nu1, nu2, fc, PHI, tau_jac):
    '''
    Set up the three layer model preconditioner matrix in Numba syntax

    Parameters
    ----------
    k,l: complex number
        wave number in dimension 0 (Fourier space)
        wave number in dimension 1 (Fourier space)
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    H1: float
        Height of the wind-farm layer
    H2: float
        Height of the upper layer
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    fc: float
        Coriolis parameter
    PHI: complex number
        Complex stratification coefficient
    tau_jac: numpy array
        Jacobians of the momentum flux terms

    Returns
    -------
    P: 2d numpy array
        Preconditioner matrix
    '''
    P = np.zeros((6, 6), dtype=np.complex128)
    sigma1 = U1*k+V1*l
    sigma2 = U2*k+V2*l
    #u1 equation
    P[0,0] += -1j*sigma1-nu1*(k**2+l**2)
    P[0,1] += +fc
    P[0,4] += -1j*k*PHI
    P[0,5] += -1j*k*PHI
    #v1 equation
    P[1,0] += -fc
    P[1,1] += -1j*sigma1-nu1*(k**2+l**2)
    P[1,4] += -1j*l*PHI
    P[1,5] += -1j*l*PHI
    #u2 equation
    P[2,2] += -1j*sigma2-nu2*(k**2+l**2)
    P[2,3] += +fc
    P[2,4] += -1j*k*PHI
    P[2,5] += -1j*k*PHI
    #v2 equation
    P[3,2] += -fc
    P[3,3] += -1j*sigma2-nu2*(k**2+l**2)
    P[3,4] += -1j*l*PHI
    P[3,5] += -1j*l*PHI
    # Add vertical momentum flux terms
    P[:4, :] += tau_jac
    #eta1 equation
    #General case
    P[4,0] += H1*k
    P[4,1] += H1*l
    P[4,4] += sigma1
    #eta2 equation
    #General case
    P[5,2] += H2*k
    P[5,3] += H2*l
    P[5,5] += sigma2
    #k=l=0
    # if k==0 and l == 0:
    if PHI == 0. or (sigma1 == 0. and (k == 0. or l == 0.)):
        P[4,4] = 1.+0.j
    if PHI == 0. or (sigma2 == 0. and (k == 0. or l == 0.)):
        P[5,5] = 1.+0.j
    return P


















