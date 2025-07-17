
'''
Piece-wise constant module for wave patterns in nonuniform atmospheres.
'''

__author__ = "Koen Devesse"
__date__ = "March 7, 2019"

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from cmath import sqrt
from numba import njit


@njit(parallel=False)
def Scorer_parameter(N, U, d2U):
    """
    Calculates the Scorer parameter for 1D flow

    :param N: Brunt-Vaisala frequency
    :param U: Wind speed in the x-direction
    :param d2U: The second derivative of the wind speed in the x-direction with respect to z
    :return: 	m^2 as given by the dispersion relation
    """
    return N ** 2 / U ** 2 - d2U / U


@njit(parallel=False)
def m_squared(N, U, V, d2U, d2V, K, L):
    """
    Calculates m^2 according to the dispersion relation.

    :param N: Brunt-Vaisala frequency
    :param U: Wind speed in the x-direction
    :param V: Wind speed in the y-direction
    :param d2U: The second derivative of the wind speed in the x-direction with respect to z
    :param d2V: The second derivative of the wind speed in the y-direction with respect to z
    :param K: 	Wave number in the x direction
    :param L: 	Wave number in the y direction
    :return: 	m^2 as given by the dispersion relation
    """
    Omega = - (U * K + V * L)
    return (K ** 2 + L ** 2) * ((N / Omega) ** 2 - 1.0) + (d2U * K + d2V * L) / Omega


@njit(parallel=False)
def m_squared_static(N, U, V, d2U, d2V, K, L):
    """
    Calculates m^2 according to the dispersion relation, neglecting hydrostatic effects.

    :param N: Brunt-Vaisala frequency
    :param U: Wind speed in the x-direction
    :param V: Wind speed in the y-direction
    :param d2U: The second derivative of the wind speed in the x-direction with respect to z
    :param d2V: The second derivative of the wind speed in the y-direction with respect to z
    :param K: 	Wave number in the x direction
    :param L: 	Wave number in the y direction
    :return: 	m^2 as given by the dispersion relation
    """
    Omega = - (U * K + V * L)
    return (K ** 2 + L ** 2) * ((N / Omega) ** 2) + (d2U * K + d2V * L) / Omega


@njit(parallel=False)
def phiConstant(N, u, v, d2u, d2v, h, u_int, v_int, du, dv, k, l, dynamic):
    """
    Returns the complex stratification coefficient, using sublayers with constant vertical wavenumbers.

    Parameters
    ----------
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
    k: float
            The wave-number in the x-direction
    l: float
            The wave-number in the y-direction
    dynamic: bool
            Whether dynamic effects are taken into account

    Returns
    -------
    phi: complex
        The stratification coefficients.
    """
    # Get amplitudes
    if dynamic:
        amps = amplitudes_numba(N, u, v, d2u, d2v, h, u_int, v_int, du, dv, k, l, m_squared)
        m2 = m_squared(N[0], u[0], v[0], d2u[0], d2v[0], k, l)
    else:
        amps = amplitudes_numba(N, u, v, d2u, d2v, h, u_int, v_int, du, dv, k, l, m_squared_static)
        m2 = m_squared_static(N[0], u[0], v[0], d2u[0], d2v[0], k, l)
    # Stratification coefficient
    Omega = - (u_int[0] * k + v_int[0] * l)
    phi = Omega / (k ** 2 + l ** 2) * (Omega * 1.j * sqrt(m2) * (amps[0] - amps[1]) / (amps[0] + amps[1])
                                       + (du[0] * k + dv[0] * l))
    return phi


@njit(parallel=False)
def amplitudes_numba(N, u, v, d2u, d2v, h, u_int, v_int, du, dv, k, l, m_function=m_squared):
    """
    Returns the amplitudes of the wave-like solutions of the problem, solving the matrix as a banded matrix.

    Parameters
    ----------
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
    k: float
            The wave-number in the x-direction
    l: float
            The wave-number in the y-direction
    m_function: callable (optional)
            Method used to calculate the vertical wavenumber (default: m_squared)

    Returns
    -------
    w: ndarray
        The amplitudes of the standing waves in each of the sublayers.
    """
    # Setting up matrix #
    omega = -(u*k + v*l)
    vector_length = 2 * len(omega)
    critical_level, solution_length = checkCritical(h, omega)
    A = np.zeros((solution_length, solution_length), dtype=np.complex128)
    # Lower boundary
    A[0][0] = 1.
    A[0][1] = 1.
    # Main loop
    for i in range(0, critical_level):
        omega_i = -(u_int[2*i+1]*k+v_int[2*i+1]*l)
        omega_j = -(u_int[2*i+2]*k+v_int[2*i+2]*l)
        domegadz_i = -(du[2*i+1]*k+dv[2*i+1]*l)
        domegadz_j = -(du[2*i+2]*k+dv[2*i+2]*l)
        m2_i = m_function(N[i], u[i], v[i], d2u[i], d2v[i], k, l)
        m_i = sqrt(m2_i)
        m2_j = m_function(N[i+1], u[i+1], v[i+1], d2u[i+1], d2v[i+1], k, l)
        m_j = sqrt(m2_j)
        # Kinematic condition
        A[2*i+1][2*i] = np.exp((h[i+1]-h[i])*1j*m_i)/omega_i
        A[2*i+1][2*i+1] = np.exp(-(h[i+1]-h[i])*1j*m_i)/omega_i
        A[2*i+1][2*i+2] = -1/omega_j
        A[2*i+1][2*i+3] = -1/omega_j
        # Dynamic condition
        A[2*i+2][2*i] = (omega_i*1j*m_i - domegadz_i) * np.exp((h[i+1]-h[i])*1j*m_i)
        A[2*i+2][2*i+1] = (-omega_i*1j*m_i - domegadz_i) * np.exp(-(h[i+1]-h[i])*1j*m_i)
        A[2*i+2][2*i+2] = -(omega_j*1j*m_j - domegadz_j)
        A[2*i+2][2*i+3] = -(-omega_j*1j*m_j - domegadz_j)
    # Upper boundary
    m2_upper = m_function(N[critical_level], u[critical_level], v[critical_level], 0., 0., k, l)
    if m2_upper >= 0:
        Omega = -(u[critical_level] * k + v[critical_level] * l)
        if np.sign(Omega) > 0:
            A[solution_length-1][solution_length-2] = 1.
        else:
            A[solution_length - 1][solution_length - 1] = 1.
    else:
        A[solution_length - 1][solution_length - 1] = 1.

    # Initialize matrix
    b = np.zeros(solution_length, dtype=np.complex128)

    b[0] = 1

    # Solve #
    w = np.zeros(vector_length, dtype=np.complex128)
    w[:solution_length] = np.linalg.solve(A, b)

    return w


@njit(parallel=False)
def checkCritical(h, omega):
    '''
    Check at which sublayer a critical layer occurs, and thus up to where has to be solved in a piecewise method.
    Also returns the size of the matrix in the piecewise method.
    '''
    # Main loop
    for i in range(0, len(omega)-1):
        omega_i = omega[i]
        omega_j = omega[i+1]
        if (np.sign(omega_j) != np.sign(omega_i)) or omega_j == 0.:
            return i, 2*(i+1)
    return len(omega)-1, 2*len(omega)


def basic_solver(parameters, k, l):

    # Unpack parameters
    h, N, u, v, du, dv, d2u, d2v = parameters

    # Step size
    dh = h[1] - h[0]

    # m squared
    m2 = m_squared(N, u, v, d2u, d2v, k, l)

    ### Initialise matrix inputs ###
    row_inds = []	# Row indices
    col_inds = []	# Column indices
    data = []		# The corresponding matrix elements

    # Surface boundary condition
    row_inds.extend([0])
    col_inds.extend([0])
    data.extend([1.])

    # Helmholtz equation #
    for i in range(1, h.size-2):
        row_inds.extend([i, i, i])
        col_inds.extend([i-1, i, i+1])
        data.extend([1. / dh ** 2, +m2[i] - 2. / dh ** 2, 1. / dh ** 2])
    row_inds.extend([h.size-2, h.size-2, h.size-2, h.size-2])
    col_inds.extend([h.size-4, h.size-3, h.size-2, h.size-1])
    data.extend([1. / dh ** 2, -4. / dh ** 2, 5. / dh ** 2, +m2[-1] - 2. / dh ** 2])

    # Upper boundary conditions #
    # Dynamic
    omega = -(u[-1]*k + v[-1]*l)
    row_inds.extend([h.size-1, h.size-1, h.size-1, h.size-1, h.size-1])
    col_inds.extend([h.size-3, h.size-2, h.size-1, h.size, h.size+1])
    data.extend([1. / (2.*dh), -4. / (2.*dh), 3. / (2.*dh),
                 1j*sqrt(m2[-1])*np.exp(1j*sqrt(m2[-1])),
                 -1j*sqrt(m2[-1])*np.exp(-1j*sqrt(m2[-1]))])
    # Kinematic
    omega = -(u[-1]*k + v[-1]*l)
    row_inds.extend([h.size, h.size, h.size])
    col_inds.extend([h.size-1, h.size, h.size+1])
    data.extend([1., -np.exp(1j*sqrt(m2[-1])), -np.exp(-1j*sqrt(m2[-1]))])
    # Radiation
    row_inds.extend([h.size+1])
    if omega ** 2 < N[-1] ** 2 >= 0:
        if np.sign(-(u[-1] * k + v[-1] * l)) < 0:
            col_inds.extend([h.size])
        else:
            col_inds.extend([h.size+1])
    else:
        col_inds.extend([h.size])
    data.extend([1.])

    # Initialize matrix
    A = scipy.sparse.csr_matrix((data, (row_inds, col_inds)), shape=(h.size+2, h.size+2), dtype=np.complex128)
    b = np.zeros(h.size+2, dtype=np.complex128)
    b[0] = 1.

    # Solve #
    w = scipy.sparse.linalg.spsolve(A, b)

    # Compute Phi
    omega = -(u[0]*k + v[0]*l)
    domegadz = -(du[0]*k + dv[0]*l)
    dwdz = (-3.*w[0] + 4.*w[1] - w[2]) / (2.*dh)
    # phi = omega / (k**2 + l**2) * (omega * dwdz + w[0] * domegadz)
    phi = omega / (k**2 + l**2) * (omega * dwdz)

    return phi, w[:-2]
