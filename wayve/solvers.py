#!/usr/bin/env python

"""
Contains solver routines used by the APM.
"""

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "July 18, 2022"

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt


class Solver:
    """
    Abstract class of solvers for the APM equations

    Solver objects can interact with an APM object through the methods My, Px, Ax, Bvector, and forcing.reprocess().
    With these, the solver can compute the matrix operations and right-hand side vectors associated with the APM.
    """

    def solve(self, model, verbose=False):
        """
        Solve the system of equations.

        Parameters
        ----------
        model   Model
                    Model for which the equations are solved
        verbose Boolean
                    Whether the solver prints its state frequently

        Returns
        -------
        X   np.array
                Model state
        err float
                Error
        """
        raise Exception("Solver routine not implemented!")


class NoFeedback(Solver):
    """
    Direct solve, where feedback effects are not included

    This Solver class discards the components of the APM forcing terms that depend on the APM state, so that the
    equations decouple per wave-number. This allows the APM equations to be solved through direct matrix inversion using
    the method APM.My().
    """

    def solve(self, model, verbose=False):
        """Solve the system of equations"""

        start = time.time()
        # System setup
        if verbose:
            print('Start building matrices')

        # Set up RHS
        B = model.Bvector()

        # Set up matrix inversion operator
        M = model.Moperator()

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to create matrices was {hh:02d}:{mm:02d}:{ss:02d}")

        # Solve system of equations
        if verbose:
            print('Start model calculation')
            start = time.time()

        X = M.matvec(B)

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to calculate model was {hh:02d}:{mm:02d}:{ss:02d}")

        return X, 0.


class IterativeSolver(Solver):
    """
    Abstract class of solvers using iterative procedures

    This includes both Krylov-subspace methods, which can be used for linear systems, and non-linear fixed-point
    iteration solvers.
    """

    def __init__(self, tol=1.0e-5, maxiter=10):
        """
        tol: float
            relative tolerance for the iterative solver method
        maxiter: int
            maximum number of iterations
        """
        self.__tol = tol
        self.__maxiter = maxiter

    @property
    def tol(self):
        """Relative tolerance"""
        return self.__tol

    @tol.setter
    def tol(self, value):
        self.__tol = value

    @property
    def maxiter(self):
        """Maximum number of iterations"""
        return self.__maxiter

    @maxiter.setter
    def maxiter(self, value):
        self.__maxiter = value


class Counter(object):
    '''Counter object for monitoring solver iterations'''

    def __init__(self, f_res, N, tol, disp=True):
        """Initialize a Counter object with a given residual evaluator function,
         maximum number of iterations, and tolerance."""
        self._disp = disp
        self.niter = 0
        self.counter = 0
        self.f_res = f_res
        self.N = N
        self.residu = np.array([])
        self.tol = tol
        if self._disp:
            print('1---------------------------------------{}'.format(self.N))

    def __call__(self, xk=None):
        """Update after an iteration step"""
        self.niter += 1
        self.residu = np.append(self.residu, self.f_res(xk))
        dn = round(self.niter / self.N * 40)
        ni = dn - self.counter
        if self._disp:
            for i in range(ni):
                print('|', end='', flush=True)
        self.counter = dn
        # Reached the end
        if self.niter == self.N or self.residu[self.niter-1] <= self.tol:
            if self._disp:
                print()     # Break with newline

    def plot_residual(self, method):
        """Plot residual as function of iterative solver iterations"""
        plt.figure()
        indices = np.nonzero(self.residu)[0]
        residual = self.residu[indices]
        plt.semilogy(indices+1,residual,linestyle='--',color='darkblue',marker='s',
                     markerfacecolor='white',markeredgecolor='darkblue', markeredgewidth=2.0,
                     linewidth=2, markersize=7,zorder=1)
        plt.axhline(y=self.tol,color='magenta',linestyle='--')
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.legend(('Residual',
                    'Threshold'
                    ),fontsize=12)
        plt.xlabel(method+' iteration',fontsize=14)
        plt.ylabel('Residual',fontsize=14)
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.xlim([0.9,len(indices)+0.1])
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)


class KrylovMethod(IterativeSolver):
    """Solvers using Krylov subspace methods"""

    def solve(self, model, verbose=False):
        """Solve the system of equations"""
        ##########################
        # Build A and B matrices #
        ##########################

        start = time.time()
        if verbose:
            print('Start building matrices')

        # Set up RHS
        B = model.Bvector()

        # Set up preconditioner
        M = model.Moperator()

        # Set up matrix operator
        A = model.Aoperator()

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to create matrices was {hh:02d}:{mm:02d}:{ss:02d}")
            start = time.time()

        #############################
        # Solve system of equations #
        #############################

        X, err = self.solver_call(A, B, M, verbose)

        res = np.linalg.norm(A.matvec(X)-B) / np.linalg.norm(B)

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to solve system was {hh:02d}:{mm:02d}:{ss:02d}")
            print(f'Euclidean norm of the complex residual is {res * np.linalg.norm(B)}')

        return X, res

    def solver_call(self, A, B, M, verbose):
        """
        Solve the given system of equations
        """
        if not verbose:
            X, err = self.algorithm(A, B, M, None)
        else:
            f_res = lambda xk: np.linalg.norm(A.matvec(xk) - B) / np.linalg.norm(B)
            counter = Counter(f_res, self.maxiter, self.tol, disp=True)
            X, err = self.algorithm(A, B, M, counter)
            # Plot residual
            counter.plot_residual(method=self.method_name())
            if verbose:
                print(self.method_name(), 'finished with output flag ', err)
                print(self.method_name(), 'needed', counter.niter, 'iterations')
        return X, err

    @staticmethod
    def method_name():
        raise Exception("Unnamed method")

    def algorithm(self, A, B, M, counter):
        """Call to the specific algorithm used by the Solver"""
        raise Exception("No default solver algorithm available!")


class LGMRES(KrylovMethod):
    """Class for LGMRES solvers"""

    @staticmethod
    def method_name():
        return "LGMRES"

    def algorithm(self, A, B, M, counter=None):
        return scipy.sparse.linalg.lgmres(A, B,
                                          rtol=self.tol,
                                          maxiter=self.maxiter,
                                          M=M,
                                          callback=counter
                                          )


class GCROTMK(KrylovMethod):
    """Class for GCROTMK solvers"""

    def __init__(self, tol=1.0e-10, maxiter=10, m=20, k=20):
        super().__init__(tol, maxiter)
        self.__m = m
        self.__k = k

    @property
    def m(self):
        return self.__m

    @m.setter
    def m(self, value):
        self.__m = value

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, value):
        self.__k = value

    @staticmethod
    def method_name():
        return 'GCROT(m,k)'

    def algorithm(self, A, B, M, counter=None):
        return scipy.sparse.linalg.gcrotmk(A, B,
                                           M=M,
                                           rtol=self.tol,
                                           maxiter=self.maxiter,
                                           m=self.m,
                                           k=self.k,
                                           callback=counter)


class BiCGSTAB(KrylovMethod):
    """Class for BiCGSTAB solvers"""

    @staticmethod
    def method_name():
        return 'BiCGSTAB'

    def algorithm(self, A, B, M, counter=None):
        return scipy.sparse.linalg.bicgstab(A, B,
                                            rtol=self.tol,
                                            maxiter=self.maxiter,
                                            M=M,
                                            callback=counter
                                            )


class GMRES(KrylovMethod):
    """Class for GMRES solvers"""

    def __init__(self, tol=1.0e-10, restart=100, maxiter=100000):
        super().__init__(tol, maxiter)
        self.__restart = restart

    @property
    def restart(self):
        return self.__restart

    @restart.setter
    def restart(self, value):
        self.__restart = value

    @staticmethod
    def method_name():
        return 'GMRES'

    def algorithm(self, A, B, M, counter=None):
        return scipy.sparse.linalg.gmres(A, B,
                                         rtol=self.tol,
                                         restart=self.restart,
                                         maxiter=self.maxiter,
                                         callback=counter)


class FixedPointIteration(IterativeSolver):
    """
    Basic fixed point iteration solver for the APM.

    At each iteration, the forcing terms are re-processed. This allows non-linear effects to be taken into account.
    """

    def __init__(self, tol=1.0e-10, maxiter=10, relax=1.):
        """
        relax: float
            relaxation factor
        """
        super().__init__(tol, maxiter)
        self.relax = relax

    @property
    def relax(self):
        """Relaxation factor"""
        return self.__relax

    @relax.setter
    def relax(self, value):
        if np.logical_or(value < 0., value > 1.):
            raise Exception("Relaxation factor not accepted!")
        self.__relax = value

    def solve(self, model, verbose=False):
        """Solve the system of equations"""

        start = time.time()
        if verbose:
            print(f"Setting up matrices")

        # Set up RHS
        B = model.Bvector()

        # Set up matrix inversion operator
        M = model.Moperator()

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to create matrices was {hh:02d}:{mm:02d}:{ss:02d}")
            start = time.time()

        # Initial solve with no perturbation #

        # Solve the matrix
        x = M.matvec(B)
        result = model.store_solution(x, 0.)

        if verbose:
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f'Time for initial solve was {hh:02d}:{mm:02d}:{ss:02d}')
            start = time.time()

        # Fixed point iteration #

        # Residual evaluation function
        def f_res(xk):
            # RHS and LHS
            Pk = model.Px(xk)
            Bk = model.Bvector(xk)
            # Denominator
            denom = scipy.linalg.norm(Bk)
            if denom == 0.:
                return 0.
            return scipy.linalg.norm(np.abs(Pk - Bk)) / denom

        # Counter object
        if verbose:
            counter = Counter(f_res, self.maxiter, self.tol, disp=True)

        # Update wind farm coupling with new state
        model.forcing.reprocess(model, result)

        # Start up iteration
        step = 1
        res = f_res(x)
        # Counter object
        if verbose:
            counter(x)

        # Iteration
        while res > self.tol and step < self.maxiter:
            # Set up RHS
            B = model.Bvector(x)
            # Solve the matrix
            x_solve = M.matvec(B)
            x = x + self.relax * (x_solve - x)
            # Handle results
            result = model.store_solution(x, 0.)
            # Update wind farm coupling with new state
            model.forcing.reprocess(model, result)
            # Get residual
            res = f_res(x)
            # Increase step counter
            step += 1
            # Update counter object
            if verbose:
                counter(x)

        if verbose:
            # Convergence statement
            if res <= self.tol:
                print('FPI converged after', counter.niter, 'iterations')
            else:
                print('FPI did not converge after', counter.niter, 'iterations')
            # Plot residual
            counter.plot_residual(method="fixed_point")
            # Computation time
            end = time.time()
            hh = int(end - start) // 3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Time to solve system was {hh:02d}:{mm:02d}:{ss:02d}")

        return x, res
