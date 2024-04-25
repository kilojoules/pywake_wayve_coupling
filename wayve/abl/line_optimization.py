'''
Line optimization code

Code to fit profiles using straight lines, used in determining the altitude of the tropopause.
'''

__author__ = "Sebastiaan Jamaer"

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize, Bounds, lsq_linear


# ==================================== #
#         Helper functions             #
# ==================================== #
def compute_residuals(nLines, geoheight, Tpotential_true, par):
    return Tpotential_true - theta_approximation(nLines, geoheight, par)

# Theta approximation
def theta_approximation(nLines, z, par):
    """
    These implementations of the theta-approximation are more than 10x more efficient and are the same (with the
    correct package requirements of course).
    :param nLines:
    :param z:
    :param par:
    :return:
    """
    if nLines == 2:
        return __theta_approximation_2L(z, par)
    elif nLines == 3:
        return __theta_approximation_3L(z, par)
    elif nLines == 4:
        return __theta_approximation_4L(z, par)
    elif nLines == 5:
        return __theta_approximation_5L(z, par)
    else:
        assert False, "theta approximation for number of lines is not implemented"
def __theta_approximation_2L(z, par):
    """
    This function implements the approximation (2) from the working document.
    :param z: Height as input of approximation (2)
    :param par: Parameters of the approximation (2)
        par[0]: theta_0
        par[1]: alpha_1
        par[2]: z_1
        par[3]: alpha_2
        par[4]: z_2
        par[5]: alpha_3
    :return: The value of approximation (2) given parameters par and in the point z
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    return theta_0 + alpha_1*z + alpha_2*np.maximum(z-zeta_1, 0)
def __theta_approximation_3L(z, par):
    """
    This function implements the approximation (2) from the working document.
    :param z: Height as input of approximation (2)
    :param par: Parameters of the approximation (2)
        par[0]: theta_0
        par[1]: alpha_1
        par[2]: z_1
        par[3]: alpha_2
        par[4]: z_2
        par[5]: alpha_3
    :return: The value of approximation (2) given parameters par and in the point z
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    zeta_2  = par[4]
    alpha_3 = par[5]

    return theta_0 + alpha_1*z + alpha_2*np.maximum(z-zeta_1, 0) + alpha_3*np.maximum(z-zeta_2, 0)
def __theta_approximation_4L(z, par):
    """
    This function implements the approximation (2) from the working document.
    :param z: Height as input of approximation (2)
    :param par: Parameters of the approximation (2)
        par[0]: theta_0
        par[1]: alpha_1
        par[2]: z_1
        par[3]: alpha_2
        par[4]: z_2
        par[5]: alpha_3
    :return: The value of approximation (2) given parameters par and in the point z
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1     = par[2]
    alpha_2 = par[3]
    zeta_2     = par[4]
    alpha_3 = par[5]
    zeta_3  = par[6]
    alpha_4 = par[7]
    return theta_0 + alpha_1*z + alpha_2*np.maximum(z-zeta_1, 0) + alpha_3*np.maximum(z-zeta_2, 0) + \
        alpha_4*np.maximum(z-zeta_3, 0)
def __theta_approximation_5L(z, par):
    """
    This function implements the approximation (2) from the working document.
    :param z: Height as input of approximation (2)
    :param par: Parameters of the approximation (2)
        par[0]: theta_0
        par[1]: alpha_1
        par[2]: z_1
        par[3]: alpha_2
        par[4]: z_2
        par[5]: alpha_3
    :return: The value of approximation (2) given parameters par and in the point z
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    zeta_2  = par[4]
    alpha_3 = par[5]
    zeta_3  = par[6]
    alpha_4 = par[7]
    zeta_4  = par[8]
    alpha_5 = par[9]
    return theta_0 + alpha_1*z + alpha_2*np.maximum(z-zeta_1, 0) + alpha_3*np.maximum(z-zeta_2, 0) + \
        alpha_4*np.maximum(z-zeta_3, 0) + alpha_5*np.maximum(z-zeta_4, 0)

# ========================================= #
#            compute gradient               #
# ========================================= #
def compute_gradient(nLines, geoheight, par):
    """
    Returns the gradient of the function theta - \hat{theta} given the parameters of the model.
    The resulting gradient helps the optimization problem and has dimensions nx6 where n is the number of
    pressure levels in geoheight.
    :param nLines: the number of lines in the theta-approximation
    :param geoheight:
    :param par:
    :return: Returns the Jacobian of the function theta -
    hat{theta(par)} which is a numpy array of (size(geoheight), 6)
    """
    if nLines==2:
        return compute_gradient_2line(geoheight, par)
    if nLines==3:
        return compute_gradient_3line(geoheight, par)
    if nLines==4:
        return compute_gradient_4line(geoheight, par)
    if nLines==5:
        return compute_gradient_5line(geoheight, par)
    else:
        assert False, "the gradient for the specified number of lines is not implemented"
def compute_gradient_2line(geoheight, par):
    """
    Returns the gradient of the function theta - \hat{theta} given the parameters of the model.
    The resulting gradient helps the optimization problem and has dimensions nx6 where n is the number of
    pressure levels in geoheight.
    :param geoheight:
    :param par:
    :return: Returns the Jacobian of the function theta -
    hat{theta(par)} which is a numpy array of (size(geoheight), 6)
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]

    # Compute the slabs where each part of the function is used
    slab0   = np.where(geoheight <= zeta_1)
    slab1   = np.where(geoheight > zeta_1)

    jacobian = np.zeros((np.size(geoheight), np.size(par)))

    # Next pd are from the last equation of \hat{theta} (so with all parameters)
    jacobian[:, 0]  = 1                      # partial derivatives to theta 0
    jacobian[:, 1]  = geoheight              # pd to alpha 1
    jacobian[:, 2]  = -alpha_2               # pd to zeta 1
    jacobian[:, 3]  = geoheight - zeta_1     # pd to alpha 2

    # Now set pd to zero of equations below certain zeta_i
    jacobian[slab0, 2:np.size(par)] = 0      # pd to zeta 1 ... alpha 4 are zero for points z<=zeta 1

    jacobian = -jacobian                     # compute jacobian of theta-\hat{theta}
    return jacobian
def compute_gradient_3line(geoheight, par):
    """
    Returns the gradient of the function theta - \hat{theta} given the parameters of the model.
    The resulting gradient helps the optimization problem and has dimensions nx6 where n is the number of
    pressure levels in geoheight.
    :param geoheight:
    :param par:
    :return: Returns the Jacobian of the function theta -
    hat{theta(par)} which is a numpy array of (size(geoheight), 6)
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    zeta_2  = par[4]
    alpha_3 = par[5]


    # Compute the slabs where each part of the function is used
    slab0   = np.where(geoheight <= zeta_1)
    slab1   = np.all([zeta_1 < geoheight, geoheight <= zeta_2], axis=0)
    slab2   = np.where(geoheight > zeta_2)

    jacobian = np.zeros((np.size(geoheight), np.size(par)))

    # Next pd are from the last equation of \hat{theta} (so with all parameters)
    jacobian[:, 0]  = 1                      # partial derivatives to theta 0
    jacobian[:, 1]  = geoheight              # pd to alpha 1
    jacobian[:, 2]  = -alpha_2               # pd to zeta 1
    jacobian[:, 3]  = geoheight - zeta_1     # pd to alpha 2
    jacobian[:, 4]  = -alpha_3               # pd to zeta 2
    jacobian[:, 5]  = geoheight - zeta_2     # pd to alpha 3

    # Now set pd to zero of equations below certain zeta_i
    jacobian[slab0, 2:np.size(par)] = 0      # pd to zeta 1 ... alpha 4 are zero for points z<=zeta 1
    jacobian[slab1, 4:np.size(par)] = 0      # pd to zeta 2 ... alpha 4 are zero for points z<=zeta 2

    jacobian = -jacobian                     # compute jacobian of theta-\hat{theta}
    return jacobian
def compute_gradient_4line(geoheight, par):
    """
    Returns the gradient of the function theta - \hat{theta} given the parameters of the model.
    The resulting gradient helps the optimization problem and has dimensions nx6 where n is the number of
    pressure levels in geoheight.
    :param geoheight:
    :param par:
    :return: Returns the Jacobian of the function theta -
    hat{theta(par)} which is a numpy array of (size(geoheight), 6)
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    zeta_2  = par[4]
    alpha_3 = par[5]
    zeta_3  = par[6]
    alpha_4 = par[7]

    # Compute the slabs where each part of the function is used
    slab0   = np.where(geoheight <= zeta_1)
    slab1   = np.all([zeta_1 < geoheight, geoheight <= zeta_2], axis=0)
    slab2   = np.all([zeta_2 < geoheight, geoheight <= zeta_3], axis=0)
    slab3   = np.where(geoheight > zeta_3)

    jacobian = np.zeros((np.size(geoheight), np.size(par)))

    # Next pd are from the last equation of \hat{theta} (so with all parameters)
    jacobian[:, 0]  = 1                      # partial derivatives to theta 0
    jacobian[:, 1]  = geoheight              # pd to alpha 1
    jacobian[:, 2]  = -alpha_2               # pd to zeta 1
    jacobian[:, 3]  = geoheight - zeta_1     # pd to alpha 2
    jacobian[:, 4]  = -alpha_3               # pd to zeta 2
    jacobian[:, 5]  = geoheight - zeta_2     # pd to alpha 3
    jacobian[:, 6]  = -alpha_4               # pd to zeta 3
    jacobian[:, 7]  = geoheight - zeta_3     # pd to alpha 4

    # Now set pd to zero of equations below certain zeta_i
    jacobian[slab0, 2:np.size(par)] = 0      # pd to zeta 1 ... alpha 4 are zero for points z<=zeta 1
    jacobian[slab1, 4:np.size(par)] = 0      # pd to zeta 2 ... alpha 4 are zero for points z<=zeta 2
    jacobian[slab2, 6:np.size(par)] = 0      # pd to zeta 3 ... alpha 4 are zero for points z<=zeta 3

    jacobian = -jacobian                     # compute jacobian of theta-\hat{theta}
    return jacobian
def compute_gradient_5line(geoheight, par):
    """
    Returns the gradient of the function theta - \hat{theta} given the parameters of the model.
    The resulting gradient helps the optimization problem and has dimensions nx6 where n is the number of
    pressure levels in geoheight.
    :param geoheight:
    :param par:
    :return: Returns the Jacobian of the function theta -
    hat{theta(par)} which is a numpy array of (size(geoheight), 6)
    """
    theta_0 = par[0]
    alpha_1 = par[1]
    zeta_1  = par[2]
    alpha_2 = par[3]
    zeta_2  = par[4]
    alpha_3 = par[5]
    zeta_3  = par[6]
    alpha_4 = par[7]
    zeta_4  = par[8]
    alpha_5 = par[9]

    # Compute the slabs where each part of the function is used
    slab0   = np.where(geoheight <= zeta_1)
    slab1   = np.all([zeta_1 < geoheight, geoheight <= zeta_2], axis=0)
    slab2   = np.all([zeta_2 < geoheight, geoheight <= zeta_3], axis=0)
    slab3   = np.all([zeta_3 < geoheight, geoheight <= zeta_4], axis=0)
    slab4   = np.where(geoheight > zeta_4)

    jacobian = np.zeros((np.size(geoheight), 10))

    # Next pd are from the last equation of \hat{theta} (so with all parameters)
    jacobian[:, 0]  = 1                      # partial derivatives to theta 0
    jacobian[:, 1]  = geoheight              # pd to alpha 1
    jacobian[:, 2]  = -alpha_2               # pd to zeta 1
    jacobian[:, 3]  = geoheight - zeta_1     # pd to alpha 2
    jacobian[:, 4]  = -alpha_3               # pd to zeta 2
    jacobian[:, 5]  = geoheight - zeta_2     # pd to alpha 3
    jacobian[:, 6]  = -alpha_4               # pd to zeta 3
    jacobian[:, 7]  = geoheight - zeta_3     # pd to alpha 4
    jacobian[:, 8]  = -alpha_5               # pd to zeta 4
    jacobian[:, 9]  = geoheight - zeta_4     # pd to alpha 5

    # Now set pd to zero of equations below certain zeta_i
    jacobian[slab0, 2:np.size(par)] = 0      # pd to zeta 1 ... alpha 5 are zero for points z<=zeta 1
    jacobian[slab1, 4:np.size(par)] = 0      # pd to zeta 2 ... alpha 5 are zero for points z<=zeta 2
    jacobian[slab2, 6:np.size(par)] = 0      # pd to zeta 3 ... alpha 5 are zero for points z<=zeta 3
    jacobian[slab3, 8:np.size(par)] = 0      # pd to zeta 4 ... alpha 5 are zero for points z<=zeta 4

    jacobian = -jacobian                     # compute jacobian of theta-\hat{theta}
    return jacobian


# ========================================== #
#            Optimizations                   #
# ========================================== #
# Simple least squares
def simple_least_squares(nLines, profiles, init_par, random_seed=None, verbose=0):
    """
    This function calculates the nLine-Optimization with constant bounds and the scipy-module least_squares
    Because only bounds are allowed in this module, the bound z1<z2 cannot be implemented.
    This optimization resulted in decent profiles for most. It is much faster then the smart_slsqp but less good.
    :param nLines: the number of lines in the approximation
    :param profiles: the profiles for which the approximation has to be done. list of np arrays with dimensions nx2
    where n does not have to be the same for all profiles. A profile P has its height in P[:, 0] and its physical value
    in P[:, 1]
    :param init_par: initial guess of parameters (required, most important are the heights zeta x and theta 0 and
    alpha 0
    :param random_seed: seed to set the numpy random generator, use this to create reproducible results (default None)
    :param verbose: level of verbose (0 and 1 implemented)
    :return: This function returns the parameters of the approximations
    """
    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    N = len(profiles)

    results = []
    for i in range(N):
        if abs(verbose - 1)<1e06:
            if i%100 ==0:
                print(i)
        parprofile  = profiles[i][:, 0]
        geoheight = profiles[i][:, 1]

        # Define cost function and jacobian function
        def residuals_optimization(par):
            return compute_residuals(nLines, geoheight, parprofile, par)
        def jacobian_optimization(par):
            J = compute_gradient(nLines, geoheight, par)
            # residuals = compute_residuals(geoheight, parprofile, par)
            return J

        # Solve optimization, if you want to speed it up/improve it you should include some bounds.
        ls = least_squares(residuals_optimization, init_par, jacobian_optimization)
        results.append(ls.x)
    return results

# Smart SLSQP
def smart_slsqp_optimization(nLines, profiles, init_par=None, random_seed=None, verbose=0):
    """
    This function calculates the nLine-Optimization with the self-defined semi-discrete optimization algorithm.
    For all profiles this algorithm resulted in a better approximation then with least-squares. However, when using
    more then 5 line segments this function becomes too slow. If needed it could be parralised (in principle)
    :param nLines: the number of lines in the approximation
    :param profiles: the profiles for which the approximation has to be done. list of np arrays with dimensions nx2
    where n does not have to be the same for all profiles. A profile P has its height in P[:, 0] and its physical value
    in P[:, 1]
    :param init_par: initial guess of parameters (not used, only here for consistency)
    :param random_seed: seed to set the numpy random generator, use this to create reproducible results (default None)
    :param verbose: level of verbose (0 and 1 implemented)
    :return: This function returns the parameters of the approximation
    """
    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    N = len(profiles)

    results = []
    initials = []
    for i in range(N):
        if verbose == 1:
            print(i)
        parprofile = profiles[i][:, 0]
        geoheight  = profiles[i][:, 1]
        def residuals_optimization(par):
            return compute_residuals(nLines, geoheight, parprofile, par)
        def cost_function(par):
            return 0.5*sum(residuals_optimization(par)**2)
        def jacobian_optimization(par):
            # Note that this is the jacobian of the cost function and not the residuals (as was the case for the
            # least-squares fit. Therefore we need to multiply it by the residuals the gradient with the residuals.
            J = compute_gradient(nLines, geoheight, par)
            residuals = residuals_optimization(par)
            return np.matmul(residuals, J)
        init_par    = find_smart_initial(nLines, parprofile, geoheight)

        ineq_con = get_inequality_constraints(nLines)

        # To speed it up you can increase the tolerance and possibly include bounds
        opt = minimize(cost_function, init_par, method='SLSQP', jac=jacobian_optimization, constraints=ineq_con,
                       tol=1e-16)
        # print((opt.fun - cost_function(init_par))/cost_function(init_par))
        results.append(opt.x)
        initials.append(init_par)
    return results
def get_inequality_constraints(nLines):
    if nLines == 2:
        ineq_con = {'type': 'ineq',
                    'fun':  lambda par: np.array([par[2]]),
                    'jac':  lambda par: np.array([0, 0, 1, 0])}
    elif nLines == 3:
        ineq_con = {'type': 'ineq',
                    'fun':  lambda par: np.array([par[4]-par[2]]),
                    'jac':  lambda par: np.array([0, 0, -1, 0, 1, 0])}
    elif nLines == 4:
        ineq_con = {'type': 'ineq',
                    'fun':  lambda par: np.array([par[4]-par[2],
                                                  par[6]-par[4]]),
                    'jac':  lambda par: np.array([[0, 0, -1, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, -1, 0, 1, 0]])}
    elif nLines == 5:
        ineq_con = {'type': 'ineq',
                    'fun':  lambda par: np.array([par[4]-par[2],
                                                  par[6]-par[4],
                                                  par[8]-par[6]]),
                    'jac':  lambda par: np.array([[0, 0, -1, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, -1, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, -1, 0, 1, 0]])}
    else:
        assert False, "The inequality constraints are not implemented for this number of lines"
    return ineq_con
def find_smart_initial(nLines, t, z):
    if nLines == 2:
        return __find_smart_initial_2Lines(t, z)
    elif nLines == 3:
        return __find_smart_initial_3Lines(t, z)
    elif nLines == 4:
        return __find_smart_initial_4Lines(t, z)
    elif nLines == 5:
        return __find_smart_initial_5Lines(t, z)
    else:
        assert False, "smart initial algorithm not implemented for nLines"
def __find_smart_initial_2Lines(t, z):
    cost_best = np.inf
    z1_best     = 0
    theta_best  = 0
    alpha1_best = 0
    alpha2_best = 0
    for i in range(np.size(z)-1):
        z1 = (z[i] + z[i+1])/2
        def A():
            A = np.zeros((np.size(t), 3))
            A[:, 0] = 1
            A[:, 1] = z
            A[:, 2] = np.maximum(0, z - z1)
            return A
        lsq  = lsq_linear(A=A(), b=t)
        cost = lsq.cost
        par  = lsq.x
        if cost<cost_best:
            z1_best     = z1
            theta_best  = par[0]
            alpha1_best = par[1]
            alpha2_best = par[2]
            cost_best = cost
    initial = np.array([theta_best, alpha1_best, z1_best, alpha2_best])
    return initial
def __find_smart_initial_3Lines(t, z):
    cost_best = np.inf
    z1_best     = 0
    z2_best     = 0
    theta_best  = 0
    alpha1_best = 0
    alpha2_best = 0
    alpha3_best = 0
    for i in range(np.size(z)-1):
        z1 = (z[i] + z[i+1])/2
        for j in range(i, np.size(z)-1):
            z2 = (z[j] + z[j+1])/2
            def A():
                A = np.zeros((np.size(t), 4))
                A[:, 0] = 1
                A[:, 1] = z
                A[:, 2] = np.maximum(0, z - z1)
                A[:, 3] = np.maximum(0, z - z2)
                return A
            lsq  = lsq_linear(A=A(), b=t)
            cost = lsq.cost
            par  = lsq.x
            if cost<cost_best:
                z1_best     = z1
                z2_best     = z2
                theta_best  = par[0]
                alpha1_best = par[1]
                alpha2_best = par[2]
                alpha3_best = par[3]
                cost_best = cost
    initial = np.array([theta_best, alpha1_best, z1_best, alpha2_best, z2_best, alpha3_best])
    return initial
def __find_smart_initial_4Lines(t, z):
    """
    The function find_smart_initial for 4 lines
    :param t:
    :param z:
    :return:
    """
    cost_best      = np.inf
    zeta1_best     = 0
    zeta2_best     = 0
    zeta3_best     = 0
    theta_best     = 0
    alpha1_best    = 0
    alpha2_best    = 0
    alpha3_best    = 0
    alpha4_best    = 0
    for i in range(np.size(z)-1):
        zeta1 = (z[i] + z[i+1])/2
        for j in range(i, np.size(z)-1):
            zeta2 = (z[j] + z[j+1])/2
            for k in range(j, np.size(z)-1):
                zeta3 = (z[k] + z[k+1])/2
                def A():
                    A = np.zeros((np.size(t), 5))
                    A[:, 0] = 1
                    A[:, 1] = z
                    A[:, 2] = np.maximum(0, z - zeta1)
                    A[:, 3] = np.maximum(0, z - zeta2)
                    A[:, 4] = np.maximum(0, z - zeta3)
                    return A
                lsq  = lsq_linear(A=A(), b=t)
                cost = lsq.cost
                par  = lsq.x
                if cost<cost_best:
                    zeta1_best  = zeta1
                    zeta2_best  = zeta2
                    zeta3_best  = zeta3
                    theta_best  = par[0]
                    alpha1_best = par[1]
                    alpha2_best = par[2]
                    alpha3_best = par[3]
                    alpha4_best = par[4]
                    cost_best = cost
    initial = np.array([theta_best, alpha1_best, zeta1_best, alpha2_best, zeta2_best, alpha3_best, zeta3_best,
                        alpha4_best])
    return initial
def __find_smart_initial_5Lines(t, z):
    """
    The function find_smart_initial is also dependent on the number of lines used in the fit
    :param t:
    :param z:
    :return:
    """
    cost_best      = np.inf
    zeta1_best     = 0
    zeta2_best     = 0
    zeta3_best     = 0
    zeta4_best     = 0
    theta_best     = 0
    alpha1_best    = 0
    alpha2_best    = 0
    alpha3_best    = 0
    alpha4_best    = 0
    alpha5_best    = 0
    for i in range(np.size(z)-1):
        zeta1 = (z[i] + z[i+1])/2
        for j in range(i, np.size(z)-1):
            zeta2 = (z[j] + z[j+1])/2
            for k in range(j, np.size(z)-1):
                zeta3 = (z[k] + z[k+1])/2
                for l in range(k, np.size(z) - 1):
                    zeta4 = (z[l] + z[l + 1]) / 2
                    def A():
                        A = np.zeros((np.size(t), 6))
                        A[:, 0] = 1
                        A[:, 1] = z
                        A[:, 2] = np.maximum(0, z - zeta1)
                        A[:, 3] = np.maximum(0, z - zeta2)
                        A[:, 4] = np.maximum(0, z - zeta3)
                        A[:, 5] = np.maximum(0, z - zeta4)
                        return A
                    lsq  = lsq_linear(A=A(), b=t)
                    cost = lsq.cost
                    par  = lsq.x
                    if cost<cost_best:
                        zeta1_best  = zeta1
                        zeta2_best  = zeta2
                        zeta3_best  = zeta3
                        zeta4_best  = zeta4
                        theta_best  = par[0]
                        alpha1_best = par[1]
                        alpha2_best = par[2]
                        alpha3_best = par[3]
                        alpha4_best = par[4]
                        alpha5_best = par[5]
                        cost_best = cost
    initial = np.array([theta_best, alpha1_best, zeta1_best, alpha2_best, zeta2_best, alpha3_best, zeta3_best,
                        alpha4_best, zeta4_best, alpha5_best])
    return initial
