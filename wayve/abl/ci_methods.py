# !/usr/bin/env python

"""
Functions to estimate capping inversion parameters (implementation of the SERZ and RZ functions)

Author: Dries Allaerts, Sebastiaan Jamaer
Date: May 22, 2017 (RZ model) June 2021 (bulk), May 2022 (extra tests), April, September 2023 (review + documentation)
"""


# System imports
import time

# Numerical imports
import numpy as np
from scipy.optimize import minimize, Bounds, least_squares, LinearConstraint, NonlinearConstraint


###### Constants ###########
# list of implemented conditioners
CONDITIONERS = [None, 'regularization', 'constraints', 'combined']
# Constants for the basis functions
ksi = 1.5  # from Rampanelli and Zardi 2004
C1 = 1.0 / (2 * ksi)  # from Rampanelli and Zardi 2004
C2 = 100  # from Jamaer et al. 2023

dimensionless = False  # Set the eta2 parameter to dimensionless
C2_dimensionless = 0.1  # If eta2 is dimensionless


# ------------------------------------------------------------------------ #
#  Rampanelli and Zardi functions
# ------------------------------------------------------------------------ #
def RZfit(z, th, p0=None, dh_max=None):
    '''
    Old naming convention, kept here for backwards compability.
    '''
    return RZ_fit(z, th, p0=p0, dh_max=dh_max)


def RZ_fit(z, th, p0=np.array([1, 1, 300, 1000, 100]), dh_max=None, max_nfev=1000):
    '''
    Estimate parameters of a smooth analytical curve to fit a given
    vertical potential temperature profile

    Parameters
    ----------
    z, th: numpy 1D array
        height and potential temperature profile
    p0: list of length 5
        Initial guess for [a,b,thm,l,dh]
    dh_max:
        Optional upper bound for dh
    Returns (CIdata)
    ----------------
    CIdata: dict
        capping inversion parameters
    '''
    if not dh_max:
        # parameter list: a, b, thm, l, dh
        lbound = [0., 0., 0., 0., 0.]
        ubound = [np.inf, np.inf, np.inf, np.inf, np.inf]
    else:
        lbound = [0., 0., 0., 0., 0.]
        ubound = [np.inf, np.inf, np.inf, np.inf, dh_max]
    bounds = [lbound, ubound]

    # noinspection PyArgumentList
    def _wrap_func(z, th):
        def _residuals(params):
            return RZmodel(z, *params) - th

        return _residuals

    jac = '2-point'
    func = _wrap_func(z, th)
    method = 'trf'

    t_start = time.time()
    result = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                           max_nfev=max_nfev)
    t_end = time.time()

    [a, b, thm, l, dh] = result.x
    CIData = {}
    CIData['a'] = result.x[0]
    CIData['b'] = result.x[1]
    CIData['thm'] = result.x[2]
    CIData['l'] = result.x[3]
    CIData['dh'] = result.x[4]
    CIData['thfit'] = RZmodel(z=z, a=a, b=b, thm=thm, l=l, dh=dh)
    CIData['MAE'] = np.mean(np.abs(CIData['thfit'] - th))
    CIData['MSE'] = np.mean((CIData['thfit'] - th) ** 2)
    CIData['time'] = t_end - t_start
    CIData['success'] = result.success
    CIData['status'] = result.status
    CIData['nfev'] = result.nfev

    CIData['ksi'] = ksi  # Eq.13  from Rampanelli and Zardi
    CIData['c'] = C1  # Eq.13  from Rampanelli and Zardi
    CIData['gamma'] = CIData['b'] / (CIData['c'] * CIData['dh'])  # Eq.14  from Rampanelli and Zardi
    CIData['h0'] = CIData['l'] - CIData['dh'] / 2.  # Eq.15  from Rampanelli and Zardi
    CIData['h1'] = CIData['l']
    CIData['h2'] = CIData['l'] + CIData['dh'] / 2.  # Eq.15  from Rampanelli and Zardi
    CIData['th00'] = CIData['a'] + CIData['thm'] - CIData['gamma'] * CIData['l']  # Eq.16  from Rampanelli and Zardi
    CIData['dth'] = CIData['a'] + CIData['ksi'] * CIData['b']  # Eq.17, adjusted by Jamaer S., see Jamaer et al. 2023
    CIData['dthp'] = CIData['a']  # Eq.20  from Rampanelli and Zardi
    return CIData


def RZmodel(z, a, b, thm, l, dh):
    """
    Here for consistency
    """
    return RZ_model(z, a, b, thm, l, dh)


def RZ_model(z, a, b, thm, l, dh):
    '''
    Smooth curve representing the vertical potential temperature profile
    of a neutral atmospheric boundary layer with a capping inversion
    Rampanelli & Zardi (2004)

    Parameters
    ----------
    z: numpy 1D array
        height
    a,b,thm,l,dh: float
        fitting parameters

    Returns (th)
    ------------
    th: numpy 1D array
        vertical potential temperature profile
    '''
    ksi = 1.5
    c = 1.0 / (2 * ksi)
    eta = (z - l) / (c * dh)

    f = (np.tanh(eta) + 1.0) / 2.0
    with np.errstate(over='ignore'):  # If g goes to infinite, I display an error message and I ingore it
        g = (np.log(2 * np.cosh(eta)) + eta) / 2.0
    for i in np.where(np.isinf(g)):  # when g is infinite, I replace it with this (?)
        g[i] = (np.abs(eta[i]) + eta[i]) / 2.0

    th = thm + a * f + b * g
    return th


def _get_par_from_vector_RZ(par):
    """
    Functino to extract the separate RZ parameters from a np array.
    """
    if not isinstance(par, np.ndarray):
        par = np.array(par)
    [a, b, thm, l, dh] = np.squeeze(np.hsplit(par, 5))
    return [a, b, thm, l, dh]


def physical_parameters_RZ(par):
    """
    Extract the physical parameters from the RZ fit.

    Parameters
    ----------
    par: numpy 1D or 2D array, dictionary.
        RZ fitting parameters

    Returns (physical)
    ----------------
    CIdata: dict
        physical parameters

    """
    [a, b, thm, l, dh] = _get_par_from_vector_RZ(par)
    physical = {}
    physical['thm'] = thm
    physical['blh'] = l
    physical['CIwidth'] = dh
    physical['CIstrength'] = a + b / (2 * C1)
    physical['gamma'] = b / (C1 * dh)
    return physical


# ------------------------------------------------------------------------ #
# Surface Extended Rampanelli and Zardi functions

def SERZ_fit(z, th, p0=np.array([1, 1, 0., 300, 1000, 100, .05]), initialGuess='RZ', dh_max=None,
             conditioning='constraints', reg_factor=0, part_linear=False, solver='trust-constr',
             jacobian='manual', hessian=None, options=None):
    """
    Compute the SERZ fit defined in Jamaer et al. 2023.

    :param z:  Geopotential height profile
    :param th:  Virtual potential temperature
    :param p0:  Initial parameter guess for optimization
    :param initialGuess:  if 'RZ' use Rampanelli and Zardi profile to generate initial parameters for SERZ fit
    :param dh_max:  Maximum value for CI depth
    :param conditioning:  Which conditioning is used to ensure well-conditioned optimization. Default (and recommended) 'constraints'
    :param reg_factor:  Regularization factor used when using 'regularization' conditioning. See appendix of Jamaer et al. 2023 for more information.
    :param part_linear:  Boolean indicating if part of the optimization problem should be solved linearly (depreciated, here for consistency).
    :param solver:  What solver to use in the scipy minimize function
    :param jacobian:  if 'manual', compute the jacobian in a symbolic way, otherwise use the two-point estimation. Manual is faster and more accurate.
    :param hessian:  if 'manual', compute the hessian in a symbolic way, otherwise use the three-point estimation. No significant difference in performance.
    :param options:  Other options for the scipy minimize procedure.
    :return: CIData (dictionary)
    Dictionary with fitting parameters and the derived physical quantities.
    """

    # If no extra options were given, use empty dictionary
    if options is None:
        options = {}

    # Part linear solver not implemented
    if part_linear:
        assert False, 'Partly linear solver no longer supported.'

    # Check validity of regularization factor
    assert reg_factor >= 0 or reg_factor is None, 'regularization factor should be non-negative  (>=0)'

    # Check validity of conditioner
    assert conditioning in CONDITIONERS, 'Conditioning is should be in ' + ' '.join(map(str, CONDITIONERS))

    # Setting the bounds for the complete system
    # Parameter vector has form [ a, b, c, thm, zabl, dh, alpha]
    ubound = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0]
    lbound = [0., 0., -np.inf, 0., 0., 10, 0.]

    # change upper bound dh if given
    if dh_max is not None:
        ubound[5] = dh_max

    # Load constraints
    if conditioning == 'constraints':
        constraints = get_constraints()
    else:
        constraints = ()

    # Load regularisation multiplicators
    regularisation_multiplicators = get_regularisation_multiplicators()

    # Load cost function
    cost_function = get_cost_function(z, th, reg_factor, regularisation_multiplicators)

    # Load derivatives of cost function
    if jacobian == 'manual':
        jacobian = get_Jac(z, th, reg_factor=reg_factor, reg_mult=regularisation_multiplicators)
    if hessian == 'manual':
        hessian = get_Hess(z, th, reg_factor=reg_factor, reg_mult=regularisation_multiplicators)

    # Time fitting
    t_start = time.time()

    used_p0RZ = 0
    # First compute initial guess with RZ
    if initialGuess == 'RZ':
        if 'maxiter' in options:
            max_nfev = int(options['maxiter'] / 5)
            options['maxiter'] -= max_nfev
        else:
            max_nfev = 100
        p0RZ = RZ_fit(z, th, p0=[p0[0], p0[1], p0[3], p0[4], p0[5]], dh_max=1000, max_nfev=max_nfev)
        if satisfiesConstraints(p0RZ):
            p0 = [p0RZ['a'], p0RZ['b'], p0[2], p0RZ['thm'], p0RZ['l'], p0RZ['dh'], p0[6]]
            used_p0RZ = 1
    result = minimize(cost_function, p0, args=(), method=solver, jac=jacobian, hess=hessian,
                      bounds=Bounds(lbound, ubound), constraints=constraints, tol=None, callback=None, options=options)
    t_end = time.time()

    [a, b, c, thm, l, dh, alpha] = result.x
    CIData = {}

    # SERZ fit parameters
    CIData['a'] = result.x[0]
    CIData['b'] = result.x[1]
    CIData['c'] = result.x[2]
    CIData['thm'] = result.x[3]
    CIData['l'] = result.x[4]
    CIData['dh'] = result.x[5]
    CIData['alpha'] = result.x[6]
    CIData['thfit'] = SERZ_model(z=z, a=a, b=b, c=c, thm=thm, zabl=l, dh=dh, alpha=alpha)
    CIData['MAE'] = np.mean(np.abs(CIData['thfit'] - th))
    CIData['MSE'] = np.mean((CIData['thfit'] - th) ** 2)
    CIData['time'] = t_end - t_start
    CIData['success'] = result.success
    CIData['status'] = result.status
    CIData['nfev'] = result.nfev
    CIData['used_p0RZ'] = used_p0RZ

    # Derived physical quantities
    phys = physical_parameters([a, b, c, thm, l, dh, alpha])

    CIData['ksi'] = ksi  # Eq.13  from Rampanelli and Zardi
    CIData['c'] = C1  # Eq.13  from Rampanelli and Zardi
    CIData['gamma'] = phys['gamma']  # Eq.14  from Rampanelli and Zardi
    CIData['h0'] = CIData['l'] - CIData['dh'] / 2.  # Eq.15  from Rampanelli and Zardi
    CIData['h1'] = CIData['l']
    CIData['h2'] = CIData['l'] + CIData['dh'] / 2.  # Eq.15  from Rampanelli and Zardi
    CIData['dth'] = CIData['a'] + CIData['ksi'] * CIData['b']  # Eq.17, adjusted by Jamaer S., see Jamaer et al. 2023
    CIData['dthp'] = CIData['a']  # Eq.20  from Rampanelli and Zardi
    CIData['sl_height'] = phys['SLheight']
    CIData['slstrength'] = phys['SLstrength']
    return CIData


def SERZ_model(z, a, b, c, thm, zabl, dh, alpha):
    """
    Smooth curve representing the vertical potential temperature profile
    of a neutral atmospheric boundary layer with a capping inversion
    Rampanelli & Zardi (2004) extended with the surface layer function.

    Parameters
    ----------
    z: numpy 1D array
        height
    a,b,c,thm,zabl,dh,alpha: float
        fitting parameters

    Returns (th)
    ------------
    th: numpy 1D array
        vertical potential temperature profile
    """

    _eta1 = eta1(z, zabl, dh)
    _eta2 = eta2(z, zabl, alpha)

    _f = f(_eta1)
    _g = g(_eta1)
    _h = h(_eta2)

    th = thm + a * _f + b * _g + c * _h
    return th


def _get_par_from_vector(par):
    if isinstance(par, dict):
        try:
            return [par['a'], par['b'], par['c'], par['thm'], par['l'], par['dh'], par['alpha']]
        except:
            assert False, 'Dictionary does not contain the required keys.'
    elif not isinstance(par, np.ndarray):
        try:
            par = np.array(par)
        except:
            assert False, 'par should be either an list/array or a dictionary'
    [a, b, c, thm, l, dh, alpha] = np.squeeze(np.hsplit(par, 7))
    return [a, b, c, thm, l, dh, alpha]


def physical_parameters(par, version=1, threshold=0.5):
    """
    Get the parameters from the RZSE fit and return the physical parameters of the profile
    :param par: list = [a, b, c, thm, l, dh, alpha]
    :return: dictionary with
    'thm' = temperature shift of profile (thm)
    'BLheight' = boundary layer height (l)
    'CIwidth' = capping inversion width (dh)
    'CIstrength' = capping inversion strength (dtheta)
    'gamma' = free lapse rate
    'SLheight' = surface layer height
    'SLstrength' = surface layer strength
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    physical = {}
    physical['thm'] = thm
    physical['blh'] = l
    physical['CIwidth'] = dh
    physical['CIstrength'] = a + b / (2 * C1)
    physical['gamma'] = b / (C1 * dh)
    if version == 1:
        physical['SLheight'] = dh2_v1(par, t=threshold)
        physical['SLstrength'] = dtheta2_v1(par)
    elif version == 2:
        physical['SLheight'] = dh2_v2(par)
        physical['SLstrength'] = dtheta2_v2(par)
    return physical


# ------------------------------------------------------------------------ #
#  Surface Extended Rampanelli and Zardi helper functions
# ------------------------------------------------------------------------ #
####### Basis functions ########
### Definition ###
def eta1(z, zabl, dh):
    """
    Dimensionless height coordinate eta1 as defined in RZ and Jamaer et al. 2023

    :param z: Geopotential height  (float, nd.array)
    :param zabl: Height of the ABL  (float)
    :param dh: Width of the CI  (float)
    :return: Dimensionless height coordinate eta1
    """
    _eta1 = (z - zabl) / (C1 * dh)
    return _eta1


def eta2(z, zabl, alpha, dimensionless=False):
    """
    Height coordinate eta2. Two different implementations are possible: the dimensionless
    implementation uses C2_dimensionless*zabl to scale te height (a 'dimensionless C2'). The other
    (and default) implementation just takes C2=100.

    Experiments on 2020 indicated that the difference between the resulting values is only small.
    Approximately the same clusters and cluster fingerprints are obtained through both implementations.

    :param z:
    :param zabl:
    :param alpha:
    :param dimensionless:
    :param C2:
    :param C2_dimensionless:
    :return:
    """
    if dimensionless:
        _eta2 = (z - alpha * zabl) / (C2_dimensionless * zabl)
    else:
        _eta2 = (z - alpha * zabl) / C2
    return _eta2


def f(eta1):
    """
    Definition of the function f from Rampanelli and Zardi 2004.
    :param eta1: The scaled height coordinate eta1  (float, nd.array)
    :return: value for f
    """
    _f = (np.tanh(eta1) + 1.0) / 2.0
    return _f


def g(eta1):
    """
    Definition of the function g of SERZ.

    :param eta1: The scaled height coordinate eta1  (float, nd.array)
    :return: value for g
    """
    with np.errstate(over='ignore'):  # Overflow can occur here, I display an error message and I ingore it
        _g = (np.log(2 * np.cosh(eta1)) + eta1) / 2.0
    for i in np.where(np.isinf(_g)):  # when g is infinite, I replace it with this first order approximation
        _g[i] = (np.abs(eta1[i]) + eta1[i]) / 2.0
    return _g


def h(eta2):
    """
    Definition of the function h of SERZ.

    :param eta2: The scaled height coordinate  (float, nd.array)
    :return: value for g
    """
    with np.errstate(over='ignore'):  # Overflor can occur here, display an error message but ignore
        _h = (eta2 - np.log(np.exp(eta2) + np.exp(-eta2))) / 2
    if type(eta2) == np.float64:  # When h is infinite, replace it with first order approximation
        if np.isinf(_h):
            _h = (eta2 - np.abs(eta2)) / 2
    elif len(eta2) > 1:
        for i in np.where(np.isinf(_h)):
            _h[i] = (eta2[i] - np.abs(eta2[i])) / 2
    else:
        if np.isinf(_h):  # TODO: seems like we have some redundancy in code here.
            _h = (eta2 - np.abs(eta2)) / 2
    return _h


###### Derivatives #######
### Derivative basis funcions ###
# First order derivatives
def dfde1(eta1):
    """
    Derivative of f with respect to eta1.

    :param eta1: values for which to compute it  (float, nd.array)
    :return: Derivative  (float, nd.array)
    """
    return (1 - np.tanh(eta1) ** 2) / 2


def dgde1(eta1):
    """
    Derivative of g with respect to eta1.

    :param eta1: values for which to compute it  (float, nd.array)
    :return: Derivative  (float, nd.array)
    """
    return (np.tanh(eta1) + 1) / 2


def dhde2(eta2):
    """
    Derivative of h with respect to eta2.

    :param eta2: values for which to compute it  (float, nd.array)
    :return: Derivative  (float, nd.array)
    """
    return (1 - np.tanh(eta2)) / 2


# Second order derivatives
def ddfde1de1(eta1):
    """
    Second order derivative of f with respect to eta1.

    :param eta1: Values for which to compute the derivative  (float, np.array)
    :return: second order derivative  (float, np.array)
    """
    return -np.tanh(eta1) * (1 - np.tanh(eta1) ** 2)


def ddgde1de1(eta1):
    """
    Second order derivative of h with respect to eta1.

    :param eta1: Values for which to compute the derivative  (float, np.array)
    :return: second order derivative  (float, np.array)
    """
    return (1 - np.tanh(eta1) ** 2) / 2


def ddhde2de2(eta2):
    """
    Second order derivative of h with respect to eta2.

    :param eta2: Values for which to compute the derivative  (float, np.array)
    :return: second order derivative of h
    """
    return (np.tanh(eta2) ** 2 - 1) / 2


### Derivative height coordinates ###
# First order derivatives (all)
def de1da(z, zabl, dh):
    return 0.


def de1db(z, zabl, dh):
    return 0.


def de1dc(z, zabl, dh):
    return 0.


def de1dthm(z, zabl, dh):
    return 0.


def de1dzabl(z, zabl, dh):
    return -1 / (C1 * dh)


def de1ddh(z, zabl, dh):
    return (zabl - z) / (C1 * dh ** 2)


def de1dalpha(z, zabl, dh):
    return 0.


def de2da(z, zabl, alpha):
    return 0.


def de2db(z, zabl, alpha):
    return 0.


def de2dc(z, zabl, alpha):
    return 0.


def de2dthm(z, zabl, alpha):
    return 0.


def de2dzabl(z, zabl, alpha):
    if not dimensionless:
        return -alpha / C2
    else:
        return -1 / (C2 * zabl ** 2)


def de2ddh(z, zabl, alpha):
    return 0.


def de2dalpha(z, zabl, alpha):
    if not dimensionless:
        return -zabl / C2
    else:
        return -1. / C2


# Second order derivatives (non-zero)
def dde1dzablddh(z, zabl, dh):
    return 1 / (C1 * dh ** 2)


def dde1ddhddh(z, zabl, dh):
    return 2 * (z - zabl) / (C1 * dh ** 3)


def dde2dzabldzabl(z, zabl, alpha):
    if not dimensionless:
        return 0.
    else:
        return 2 / (C2 * zabl ** 3)


def dde2dzabldalpha(z, zabl, alpha):
    if not dimensionless:
        return -1 / C1
    else:
        return 0.


#### SERZ ####
# Model
def dh2_v1(par, t=0.5):
    """
    First version compute the height of the surface layer. Version that is used in the paper.
    :param par: parameter list of the SERZ parameters
    :param t: treshold Tr from eq. 10
    :return:
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    if np.size(c) > 1:
        c[np.where(np.abs(c) < 1e-2)] = 1e-2  # For stability
    elif np.abs(c) < 1e-2:
        c = 1e-2
    k = np.abs(t / c)
    return np.clip(alpha * l - C2 / 2 * np.log(np.exp(2 * k) - 1), a_min=0, a_max=None)


def dh2_v2(par, p=0.1):
    """
    Second version to compute height of surface layer. Gives similar results as v1.
    This version is based on a relative temperature increase compared to an absolute in v1.
    Surface layer is defined when (1-p) of the temperature change below the CI is reached.
    (so always surface layer, also if only very small gradient in ABL itself)
    :param par:
    :param p:
    :return:
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    _eta0 = -alpha * l / C2
    return alpha * l + C2 / 2 * np.log(1 / ((1 + np.exp(-2 * _eta0)) ** p - 1))


def dtheta2_v1(par):
    """
    Temperature jump (eq.12) first version. Used in paper.
    :param par:
    :return:
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    _eta0 = eta2(0, l, alpha)
    return c * h(_eta0)


def dtheta2_v2(par):
    """
    Second way of computing temperature. More complicated but did not give significant changes
    in results to v1, therefore not chosen for overall analysis.
    :param par:
    :return:
    """
    [a, b, c, thm, l, dh, alpha] = _get_par_from_vector(par)
    return c * np.minimum(-np.log(2) / 2, eta2(0, l, alpha))


# Gradient
def _grad_SERZ(z, par):
    """
    Gradient of the SERZ model with respect to its parameters.
    :param z: Values for which to calculate the gradient  (nd.array)
    :param par: Parameters for which to calculate the SERZ model  (list)
    :return: Value of the gradient (nd.array of size len(z) x len(par) )
    """
    [a, b, c, thm, zabl, dh, alpha] = par
    grad = np.zeros((len(z), 7))
    _eta1 = eta1(z=z, zabl=zabl, dh=dh)
    _eta2 = eta2(z=z, zabl=zabl, alpha=alpha)
    grad[:, 0] = f(_eta1)
    grad[:, 1] = g(_eta1)
    grad[:, 2] = h(_eta2)
    grad[:, 3] = 1

    _temp1 = a * dfde1(_eta1) + b * dgde1(_eta1)
    _temp2 = c * dhde2(_eta2)

    grad[:, 4] = de1dzabl(z=z, zabl=zabl, dh=dh) * _temp1 + de2dzabl(z=z, zabl=zabl, alpha=alpha) * _temp2
    grad[:, 5] = de1ddh(z=z, zabl=zabl, dh=dh) * _temp1 + de2ddh(z=z, zabl=zabl, alpha=alpha) * _temp2
    grad[:, 6] = de1dalpha(z=z, zabl=zabl, dh=dh) * _temp1 + de2dalpha(z=z, zabl=zabl, alpha=alpha) * _temp2
    return grad


# Hessian
def _hess_SERZ(z, par):
    """
    Hessian of the SERZ model with respect to its parameters
    :param z: Heights to compute the hessian (nd.array)
    :param par: Values of the parameters (list)
    :return: values of the hessian (nd.array of len(z) x len(par) x len(par) ).
    """
    [a, b, c, thm, zabl, dh, alpha] = par
    _eta1 = eta1(z, zabl, dh)
    _eta2 = eta2(z, zabl, alpha)

    hess = np.zeros(shape=(len(z), len(par), len(par)))
    hess[0:4, 0:4] = 0  # (da or db or dc or dthm)(da or db or dc or dthm)

    # temporary save partial derivatives for efficiency
    _dfde1 = dfde1(_eta1)
    _dgde1 = dgde1(_eta1)
    _dhde2 = dhde2(_eta2)
    _de1dl = de1dzabl(z, zabl, dh)
    _de1ddh = de1ddh(z, zabl, dh)
    _de2dl = de2dzabl(z, zabl, alpha)
    _de2dalpha = de2dalpha(z, zabl, alpha)

    _ddfde1de1 = ddfde1de1(_eta1)
    _ddgde1de1 = ddgde1de1(_eta1)
    _ddhde2de2 = ddhde2de2(_eta2)

    _dde1dlddh = dde1dzablddh(z, zabl, dh)
    _dde1ddhddh = dde1ddhddh(z, zabl, dh)
    _dde2dzabldzabl = dde2dzabldzabl(z, zabl, alpha)
    _dde2dzabldalpha = dde2dzabldalpha(z, zabl, alpha)

    # Only write out non-zero entries
    hess[:, 4, 0] = _dfde1 * _de1dl  # dadl
    hess[:, 0, 4] = hess[:, 4, 0]

    hess[:, 5, 0] = _dfde1 * _de1ddh  # daddh
    hess[:, 0, 5] = hess[:, 5, 0]

    hess[:, 4, 1] = _dgde1 * _de1dl  # dbdl
    hess[:, 1, 4] = hess[:, 4, 1]

    hess[:, 5, 1] = _dgde1 * _de1ddh  # dbddh
    hess[:, 1, 5] = hess[:, 5, 1]

    hess[:, 4, 2] = _dhde2 * _de2dl  # dcdl
    hess[:, 2, 4] = hess[:, 4, 2]

    hess[:, 6, 2] = _dhde2 * _de2dalpha  # dcdalpha
    hess[:, 2, 6] = hess[:, 6, 2]

    _temp1 = a * _dfde1 + b * _dgde1
    _temp2 = a * _ddfde1de1 + b * _ddgde1de1

    if dimensionless:
        hess[:, 4, 4] = _temp2 * _de1dl ** 2 + c * (_ddhde2de2 * _de2dl ** 2 + _dhde2 * _dde2dzabldzabl)  # dldl
    else:
        hess[:, 4, 4] = _temp2 * _de1dl ** 2 + c * _ddhde2de2 * _de2dl ** 2  # dldl

    hess[:, 4, 5] = _temp1 * _dde1dlddh + _temp2 * _de1dl * _de1ddh  # dlddh
    hess[:, 5, 4] = hess[:, 4, 5]

    if dimensionless:
        hess[:, 4, 6] = c * (_ddhde2de2 * _de2dl * _de2dalpha)  # dldalpha
    else:
        hess[:, 4, 6] = c * (_dhde2 * _dde2dzabldalpha + _ddhde2de2 * _de2dl * _de2dalpha)  # dldalpha
    hess[:, 6, 4] = hess[:, 4, 6]

    hess[:, 5, 5] = _temp1 * _dde1ddhddh + _temp2 * _de1ddh ** 2  # ddhddh

    hess[:, 6, 6] = c * (
                _ddhde2de2 * _de2dalpha ** 2)  # dalphadalpha  TODO: check this expression for dimensionless case
    return hess


####### Optimization helpers #######
#### SERZ ####
# Definition of constraints
def get_constraints(max_height_CI=4000):
    """
    Create the list of constraints for the scipy optimization for the SERZ fits.
    :param max_height_CI: The maximum value of the top of the CI.
    :return: list of contraint objects
    """
    constraints = []

    # l-dh/2>alpha*l, the bottom of the CI is higher than the top of the surface layer
    con = lambda x: x[4] - x[5] / 2 - x[6] * x[4]
    jac = lambda x: np.array([0, 0, 0, 0, 1 - x[6], -1 / 2, -x[4]])

    def hess(x, v):
        """
        The hessian of np.dot(con*v) where v is a lagrangian multiplier (since con has dim 1, just a real number)
        :param x: array of parameters [a, b, c, thm, zabl, dh, alpha]
        :param v: lagrangian multiplier
        :return: hessian of np.dot(con*v) with derivatives to the parameters
        """
        hessian = np.zeros(shape=(7, 7))
        hessian[6, 4] = -v
        hessian[4, 6] = -v
        return hessian

    constraints.append(NonlinearConstraint(con, 0, np.inf, jac=jac, hess=hess))

    # zabl+dh/2<4000, the top of the CI is lower than 4000 m
    _A = [0, 0, 0, 0, 1, 1 / 2, 0]
    constraints.append(LinearConstraint(_A, -np.inf, max_height_CI))

    return constraints


def satisfiesConstraints(p0RZ, max_height_CI=4000, min_height_SL=10):
    if p0RZ['l'] - p0RZ['dh'] / 2 > min_height_SL and p0RZ['l'] + p0RZ['dh'] / 2 < max_height_CI:
        return True
    return False


# Definition of cost function
def get_cost_function(z, th, reg_factor, reg_mult):
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])

    def cost_function(par):
        [a, b, c, thm, l, dh, alpha] = par
        th_approximation = SERZ_model(z, a=a, b=b, c=c, thm=thm, zabl=l, dh=dh, alpha=alpha)
        cost = np.mean((th - th_approximation) ** 2) + reg_factor * np.sum((par * multiplicator_list) ** 2)
        return cost

    return cost_function


# Jacobian of cost function
def get_Jac(z, th, reg_factor, reg_mult):
    """
    Define the jacobian of the SERZ optimization cost function.
    :param z: Heights for which there is a temperature
    :param th: values of the temperature at the specified heights
    :param reg_factor: regularization factor  (e.g. general scaling of the regularization)
    :param reg_mult: regularization multiplicators  (e.g. specific scaling of regularization), this should be a
    dictionary with the reg multiplicators for each individual parameter, so it should have the keys a, b, ...
    :return: The function handle to compute the jacobian. The function itself takes a set of parameters as inputs.
    """
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])

    def jacobian(par):
        [a, b, c, thm, l, dh, alpha] = par
        residuals = th - SERZ_model(z, a, b, c, thm, l, dh, alpha)
        N = len(th)

        return -2 / N * residuals @ _grad_SERZ(z, par) + 2 * reg_factor * par * multiplicator_list ** 2

    return jacobian


# Hessian of cost function
def get_Hess(z, th, reg_factor, reg_mult):
    """
    Define the hessian of the SERZ optimization cost function.
    :param z: Heights for which there is a temperature
    :param th: values of the temperature at the specified heights
    :param reg_factor: regularization factor  (e.g. general scaling of the regularization)
    :param reg_mult: regularization multiplicators  (e.g. specific scaling of regularization), this should be a
    dictionary with the reg multiplicators for each individual parameter, so it should have the keys a, b, ...
    :return: The function handle to compute the jacobian. The function itself takes a set of parameters as inputs.
    """
    multiplicator_list = np.array([reg_mult['a'], reg_mult['b'], reg_mult['c'], reg_mult['thm'], reg_mult['l'],
                                   reg_mult['dh'], reg_mult['alpha']])

    def hessian(par):
        [a, b, c, thm, l, dh, alpha] = par
        N = len(th)
        residuals = th - SERZ_model(z, a, b, c, thm, l, dh, alpha)
        grad_rzse = _grad_SERZ(z, par)
        return -2 / N * (np.tensordot(residuals, _hess_SERZ(z, par), axes=(0, 0)) -
                         np.tensordot(grad_rzse, grad_rzse, axes=(0, 0))) \
               + reg_factor * np.diag(multiplicator_list ** 2)

    return hessian


# Definition of regularisation multiplicators
def get_regularisation_multiplicators():
    """
    Return the regularisation multiplicators. These multiplicators are the inverse of the regularization weights. Taking
    the inverse allows us to ignore dealing with infinity for some weights
    :return: dictionary with the inverse weights for each parameter in the RZSE model
    """
    multiplicators = {}
    multiplicators['a'] = 1 / 5  # Kelvin^-1
    multiplicators['b'] = 0
    multiplicators['c'] = 0
    multiplicators['thm'] = 0
    multiplicators['l'] = 1 / 1000  # meter^-1
    multiplicators['dh'] = 1 / 100  # meter^-1
    multiplicators['alpha'] = 1 / 0.2  # meter/meter
    return multiplicators

