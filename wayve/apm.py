#!/usr/bin/env python

'''
Atmospheric Perturbation Model module
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "June 15, 2017"

import os
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
from wayve import apm_tools


class APM:
    '''
    Two-dimensional atmospheric perturbation model (APM)

    This class forms the core of the WAYVE framework. Various aspects of the APM, such as the numerical grid, the
    background atmospheric state, or the perturbing force, are included as attributes of this APM class. This allows
    them to operate (fairly) modularly.

    With the help of these components, the APM class defines various methods that implement various matrix operations
    related to the APM equations. This way, it offers a straightforward interface to Solver objects. Additionally, the
    class has several methods dealing with post-processing.
    '''

    def __init__(self, grid, forcing, abl, mfp, pressure):
        '''
        Parameters
        ----------
        grid: Grid object
            numerical grid
        forcing: ForcingTerm object (defined in apm_forcing.py)
            perturbing force
        abl: ABL object (defined in abl.py)
            atmospheric state
        mfp: MFP object (defined in momentum_flux_parametrizations.py)
            momentum flux parametrization
        pressure: PressureParametrization object (defined in pressure_parametrizations.py)
            pressure feedback parametrization
        '''
        self.__grid = grid
        self.__forcing = forcing
        self.__abl = abl
        self.__mfp = mfp
        self.__pressure = pressure

    def solve(self, method, verbose=False):
        '''
        Solve the atmospheric perturbation model equations

        The solver procedure consists of three basic steps:
            - Pre-processing: all necessary initial calculations are performed, so that the operations My, Px, Ax,
                Bvector, and forcing.reprocess() can be called by the Solver object.
            - Solver: the given Solver object solves the equations.
            - Post-processing: based on the state array outputted by the solver, all the relevant real perturbation
                fields and pressure perturbations are computed.

        Parameters
        ----------
        method: Solver (defined in solvers.py)
            method used to solve the system of equations
        verbose (optional): bool
            flag for printing solver results (default: False)
        '''

        self.set_up_solve(verbose)

        X, err = method.solve(self, verbose)

        result = self.store_solution(X, err, verbose=verbose)

        return result

    def set_up_solve(self, verbose=False):
        '''
        Perform pre-processing for a model run.

        Parameters
        ----------
        verbose (optional): bool
            flag for printing output (default: False)
        '''
        start = time.time()
        if verbose:
            print('Start calculating stratification coefficients Phi')
        self.pressure.preprocess(self.abl, self.grid)
        end = time.time()
        if verbose:
            hh = int(end-start)//3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f"Phi calculation took {hh:02d}:{mm:02d}:{ss:02d}")
            print('Start preprocessing model forces')
        start = time.time()
        self.forcing.preprocess(self)
        end = time.time()
        if verbose:
            hh = int(end-start)//3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f'Preprocessing time was {hh:02d}:{mm:02d}:{ss:02d}')

    def store_solution(self, X, err, verbose=False):
        '''
        Read out the given APM state and error into a dictionary.

        Parameters
        ----------
        X: numpy array
            APM state vector
        err: float or array-like
            Solver flag
        verbose (optional): bool
            flag for printing output (default: False)
        '''
        # Initialize dictionary object with error and state array
        result = {'err': err, 'x': X}
        # Store state array
        # Get relevant variables
        start = time.time()
        if verbose:
            print('Start inverse FFT')
        u1c, v1c, u2c, v2c, eta1c, eta2c = self.expandX(X)
        p1c, p2c, pc = self.format_pressure(X)
        etac = eta1c + eta2c + self.forcing.eta_c(self.grid)
        # Store Fourier-space output
        result['u1c'] = u1c
        result['v1c'] = v1c
        result['u2c'] = u2c
        result['v2c'] = v2c
        result['etac'] = etac
        result['eta1c'] = eta1c
        result['eta2c'] = eta2c
        result['pc'] = pc
        result['p1c'] = p1c
        result['p2c'] = p2c
        end = time.time()
        if verbose:
            hh = int(end-start)//3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f'Time to read out APM state was {hh:02d}:{mm:02d}:{ss:02d}')
        # Store real-space output
        start = time.time()
        if verbose:
            print('Start inverse FFT')
        result['u1r'], err_u1r = self.grid.c2r(result['u1c'], True)
        result['v1r'], err_v1r = self.grid.c2r(result['v1c'], True)
        result['u2r'], err_u2r = self.grid.c2r(result['u2c'], True)
        result['v2r'], err_v2r = self.grid.c2r(result['v2c'], True)
        result['etar'], err_etar = self.grid.c2r(result['etac'], True)
        result['eta1r'], err_pr = self.grid.c2r(result['eta1c'], True)
        result['eta2r'], err_pr = self.grid.c2r(result['eta2c'], True)
        result['pr'], err_pr = self.grid.c2r(result['pc'], True)
        result['p1r'], err_pr = self.grid.c2r(result['p1c'], True)
        result['p2r'], err_pr = self.grid.c2r(result['p2c'], True)
        end = time.time()
        if verbose:
            hh = int(end-start)//3600
            mm = int(end-start-3600*hh)//60
            ss = int(end-start-3600*hh-60*mm)
            print(f'Time to compute inverse FFT was {hh:02d}:{mm:02d}:{ss:02d}')
        return result

    def expandX(self,X):
        '''
        Expand the general solution vector into dependent variables of the problem

        Parameters
        ----------
        X: 1d numpy array
            general solution vector

        Returns
        -------
        u1,v1,u2,v2,e1,e2: 2d numpy array
            perturbation velocities and displacements in wind-farm and upper layer
        '''
        N = self.grid.N2
        u1 = X[0:N].reshape(self.grid.shape2)
        v1 = X[1*N:2*N].reshape(self.grid.shape2)
        u2 = X[2*N:3*N].reshape(self.grid.shape2)
        v2 = X[3*N:4*N].reshape(self.grid.shape2)
        e1 = X[4*N:5*N].reshape(self.grid.shape2)
        e2 = X[5*N:6*N].reshape(self.grid.shape2)
        return u1, v1, u2, v2, e1, e2

    def format_pressure(self, X):
        '''
        Compute the total pressure in each layer based on the given solution, and the pressure at the top of the ABL.

        Parameters
        ----------
        X: Numpy array
            APM solution vector

        Returns
        -------
        p1c, p2c: 2d numpy array
            pressures
        '''
        # Read out solution array
        u1c, v1c, u2c, v2c, eta1c, eta2c = self.expandX(X)
        # Get forcing contribution
        eta0c = self.forcing.eta_c(self.grid)
        p_f1, p_f2 = self.forcing.pressure_contribution(self, X)
        p1c = self.pressure.Phi * (eta1c + eta2c) + p_f1
        p2c = self.pressure.Phi * (eta1c + eta2c) + p_f2
        pc = self.pressure.Phi * (eta0c + eta1c + eta2c)
        return p1c, p2c, pc

    def Bvector(self, x=None):
        """
        Compute right-hand side of model equations (B vector)

        Parameters
        ----------
        x   np.array
                APM state. If no state is passed, only the zeroth-order contributions are returned. Otherwise, the
                full (potentially non-linear) right-hand side is returned.

        Returns
        -------
        _: 1d numpy array
            right-hand side of model equations (Fourier space)
        """
        # Get forcing terms
        if x is None:
            B = self.forcing.ZOC(self)
        else:
            B = self.forcing.RHS(self, x)
        # Defunct modes #
        N = self.grid.N2
        for var in range(6):
            self.grid.defunct_modes(B[var*N:(var+1)*N])
        return B
        
    def Aoperator(self):
        '''
        Linear operator interface for performing matrix vector products with
        the system matrix A

        Returns
        -------
        A: LinearOperator
            interface for the system matrix A
        '''
        A = scipy.sparse.linalg.LinearOperator((6*self.grid.N2,6*self.grid.N2),
                                               matvec=self.Ax,
                                               dtype=np.complex128)
        return A
    
    def Moperator(self):
        '''
        Linear operator interface for performing matrix vector products with
        the preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)

        Returns
        -------
        M: LinearOperator
            interface for the preconditioner M
        '''
        M = scipy.sparse.linalg.LinearOperator((6*self.grid.N2,6*self.grid.N2),
                                               matvec=self.My,
                                               dtype=np.complex128)
        return M

    def My(self, y):
        """
        Compute the matrix vector product M*y where preconditioner M = inv(P) approximates inv(A), excluding the
        convolution products.
        The matrix vector product My is found by solving Px=y.

        Parameters
        ----------
        y: 1d numpy array
            input vector

        Returns
        -------
        _: 1d numpy array
            matrix vector product M*y
        """
        # Atmospheric conditions
        U1 = self.abl.U1
        V1 = self.abl.V1
        U2 = self.abl.U2
        V2 = self.abl.V2
        H1 = self.abl.H1
        H2 = self.abl.H2
        nu1 = self.abl.nu1
        nu2 = self.abl.nu2
        fc = self.abl.fc
        # Jacobians of the momentum flux terms
        tau_jac = self.mfp.mf_term_Jac(self.abl)
        # Stratification coefficients
        Phi = self.pressure.Phi
        # Numerical grid
        N = self.grid.N2
        ks = self.grid.ks2
        ls = self.grid.ls
        # Matrix operation
        M = apm_tools.evaluate_M(ks, ls, U1, V1, U2, V2, H1, H2, nu1, nu2, fc, Phi, tau_jac, N, y)
        # Defunct modes
        for var in range(6):
            self.grid.defunct_modes(M[var*N:(var+1)*N])
        return M
    
    def Px(self, x):
        '''
        Compute the matrix vector product P*x
    
        The matrix P corresponds to the one used by the preconditioner, and contains all the linear terms that decouple
        per wavenumber.
    
        Parameters
        ----------
        x: 1d numpy array
            input vector
    
        Returns
        -------
        _: 1d numpy array
            matrix vector product P*x
        '''
        # Atmospheric conditions
        U1 = self.abl.U1
        V1 = self.abl.V1
        U2 = self.abl.U2
        V2 = self.abl.V2
        H1 = self.abl.H1
        H2 = self.abl.H2
        nu1 = self.abl.nu1
        nu2 = self.abl.nu2
        fc = self.abl.fc
        # Jacobians of the momentum flux terms
        tau_jac = self.mfp.mf_term_Jac(self.abl)
        # Stratification coefficients
        Phi = self.pressure.Phi
        # Numerical grid
        N = self.grid.N2
        ks = self.grid.ks2
        ls = self.grid.ls
        # Apply matrix operation
        y = apm_tools.evaluate_P(ks, ls, U1, V1, U2, V2, H1, H2, nu1, nu2, fc, Phi, tau_jac, N, x)
        # Defunct modes
        for var in range(6):
            self.grid.defunct_modes(y[var*N:(var+1)*N])
        return y

    def Ax(self,x):
        '''
        Compute the matrix vector product A*x, which includes the linear terms that don't decouple per wavenumber.

        Parameters
        ----------
        x: 1d numpy array
            input vector

        Returns
        -------
        _: 1d numpy array
            matrix vector product A*x
        '''
        # Get P*x
        Px = self.Px(x)
        # Get FOC from forcing terms
        F = self.forcing.FOC(self, x)
        # Add FOC to Px
        y = Px - F  # Subtraction because of sign convention in code
        return y

    @staticmethod
    def save_to_file(result, folder):
        """
        Saves the given APM state as .txt files in the given folder.

        Parameters
        ----------
        result  dict
            APM state
        folder  str
            Folder where the files are stored
        """
        # Checking if the directory exist or not. If the directory is not present then create it.
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # State array
        np.savetxt(folder + 'x.txt', result['x'], fmt='%f')
        # Complex fields
        np.savetxt(folder + 'u1c.txt', result['u1c'], fmt='%f')
        np.savetxt(folder + 'v1c.txt', result['v1c'], fmt='%f')
        np.savetxt(folder + 'u2c.txt', result['u2c'], fmt='%f')
        np.savetxt(folder + 'v2c.txt', result['v2c'], fmt='%f')
        np.savetxt(folder + 'pc.txt', result['pc'], fmt='%f')
        np.savetxt(folder + 'p1c.txt', result['p1c'], fmt='%f')
        np.savetxt(folder + 'p2c.txt', result['p2c'], fmt='%f')
        np.savetxt(folder + 'etac.txt', result['etac'], fmt='%f')
        np.savetxt(folder + 'eta1c.txt', result['eta1c'], fmt='%f')
        np.savetxt(folder + 'eta2c.txt', result['eta2c'], fmt='%f')
        # Real fields
        np.savetxt(folder + 'u1r.txt', result['u1r'], fmt='%f')
        np.savetxt(folder + 'v1r.txt', result['v1r'], fmt='%f')
        np.savetxt(folder + 'u2r.txt', result['u2r'], fmt='%f')
        np.savetxt(folder + 'v2r.txt', result['v2r'], fmt='%f')
        np.savetxt(folder + 'pr.txt', result['pr'], fmt='%f')
        np.savetxt(folder + 'p1r.txt', result['p1r'], fmt='%f')
        np.savetxt(folder + 'p2r.txt', result['p2r'], fmt='%f')
        np.savetxt(folder + 'etar.txt', result['etar'], fmt='%f')
        np.savetxt(folder + 'eta1r.txt', result['eta1r'], fmt='%f')
        np.savetxt(folder + 'eta2r.txt', result['eta2r'], fmt='%f')
        return

    @staticmethod
    def read_from_file(folder):
        """
        Sets up a APM state from .txt files in the given folder.

        Parameters
        ----------
        folder  str
            Folder where the files are stored
        """
        result = {}
        # State array
        result['x'] = np.loadtxt(folder + 'x.txt', dtype=np.complex_)
        # Complex fields
        result['u1c'] = np.loadtxt(folder + 'u1c.txt', dtype=np.complex_)
        result['v1c'] = np.loadtxt(folder + 'v1c.txt', dtype=np.complex_)
        result['u2c'] = np.loadtxt(folder + 'u2c.txt', dtype=np.complex_)
        result['v2c'] = np.loadtxt(folder + 'v2c.txt', dtype=np.complex_)
        result['pc'] = np.loadtxt(folder + 'pc.txt', dtype=np.complex_)
        result['p1c'] = np.loadtxt(folder + 'p1c.txt', dtype=np.complex_)
        result['p2c'] = np.loadtxt(folder + 'p2c.txt', dtype=np.complex_)
        result['etac'] = np.loadtxt(folder + 'etac.txt', dtype=np.complex_)
        result['eta1c'] = np.loadtxt(folder + 'eta1c.txt', dtype=np.complex_)
        result['eta2c'] = np.loadtxt(folder + 'eta2c.txt', dtype=np.complex_)
        # Real fields
        result['u1r'] = np.loadtxt(folder + 'u1r.txt', dtype=float)
        result['v1r'] = np.loadtxt(folder + 'v1r.txt', dtype=float)
        result['u2r'] = np.loadtxt(folder + 'u2r.txt', dtype=float)
        result['v2r'] = np.loadtxt(folder + 'v2r.txt', dtype=float)
        result['etar'] = np.loadtxt(folder + 'etar.txt', dtype=float)
        result['eta1r'] = np.loadtxt(folder + 'eta1r.txt', dtype=float)
        result['eta2r'] = np.loadtxt(folder + 'eta2r.txt', dtype=float)
        result['pr'] = np.loadtxt(folder + 'pr.txt', dtype=float)
        result['p1r'] = np.loadtxt(folder + 'p1r.txt', dtype=float)
        result['p2r'] = np.loadtxt(folder + 'p2r.txt', dtype=float)
        return result

    @property
    def grid(self):
        '''Numerical grid'''
        return self.__grid

    @property
    def abl(self):
        '''Atmospheric state'''
        return self.__abl

    @property
    def forcing(self):
        '''Perturbing force'''
        return self.__forcing

    @property
    def mfp(self):
        '''Momentum flux parametrization'''
        return self.__mfp

    @property
    def pressure(self):
        '''Pressure feedback parametrization'''
        return self.__pressure
