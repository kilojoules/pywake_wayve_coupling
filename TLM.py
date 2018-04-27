#!/usr/bin/env python

'''
Three-layer module

Module defining data structures for
- Numerical grids
- Atmospheric state
- Three-layer module
'''
__author__ = "Dries Allaerts"
__date__ = "June 15, 2017"

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.linalg
import scipy.sparse.linalg
import mpmath
import os
import time
from py4sp import loadsp
from py4sp import mypy
from py4sp import CIops

class Grid(object):
    '''
    Generic grid object
    
    Data structure containing information about the numerical domain
    and the discretization
    '''
    def __init__(self):
        self.__N = None
        self.__shape = None

    @property
    def N(self):
        '''Total number of grid points'''
        return self.__N
    @N.setter
    def N(self,value):
        self.__N = value
    @property
    def shape(self):
        '''Dimensions of the numerical domain'''
        return self.__shape
    @shape.setter
    def shape(self,value):
        self.__shape = value

class Stat1Dgrid(Grid):
    '''
    Grid for one-dimensional, stationary simulation
    '''
    def __init__(self,Lx,Nx):
        '''
        Parameters
        ----------
        Lx: float
            Length of the numerical domain
        Nx: int
            Number of grid points
        '''
        super().__init__()
        assert np.mod(Nx,2)==0, 'Error, current implementation only allows even Nx'
            #Don't allow uneven grid sizes,
            #it is implicitly assumed that there are defunct modes

        self.__Lx = Lx
        self.__Nx = Nx
        self.N = self.Nx
        self.shape = (self.Nx,)

    def deal_grid(self):
        '''
        Dealiasing grid

        Returns
        -------
        _: Stat1Dgrid
            Dealiasing grid with size = 3/2 original grid size
        '''
        assert np.mod(self.Nx,4)==0, 'Error, Nx is not a multiple of 4 so dealiasing grid is uneven'
            #Make sure dealising grid has even number of grid point,
        return Stat1Dgrid(self.Lx,int(3*self.Nx/2))

    @property
    def Lx(self):
        '''Length of the numerical domain in dimension 0'''
        return self.__Lx
    @property
    def Nx(self):
        '''Number of grid points in dimension 0'''
        return self.__Nx
    @property
    def dx(self):
        '''Grid size in dimension 0'''
        return self.Lx/self.Nx
    @property
    def xs(self):
        '''Array with grid points in dimension 0 (real space)'''
        return np.linspace(0,self.Lx,self.Nx,endpoint=False)
    @property
    def ks(self):
        '''Array with wave numbers in dimension 0 (Fourier space)'''
        #Use built-in function fftfreq, which works for both even and uneven Nx
        #Also shift wavenumbers so that k=0 is at Nx/2
        return 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(self.Nx,self.dx))

class Stat2Dgrid(Grid):
    '''
    Grid for two-dimensional, stationary simulation
    '''
    def __init__(self,Lx,Nx,Ly,Ny):
        '''
        Parameters
        ----------
        Lx,Ly: float
            Length of the numerical domain (dimension 0 and 1)
        Nx,Ny: int
            Number of grid points (dimension 0 and 1)
        '''
        super().__init__()
        assert np.mod(Nx,2)==0, 'Error, current implementation only allows even Nx'
        assert np.mod(Ny,2)==0, 'Error, current implementation only allows even Ny'
            #Don't allow uneven grid sizes,
            #it is implicitly assumed that there are defunct modes

        self.__Lx = Lx
        self.__Nx = Nx
        self.__Ly = Ly
        self.__Ny = Ny
        self.N = self.Nx*self.Ny
        self.shape = (self.Nx,self.Ny)

    def deal_grid(self):
        '''
        Dealiasing grid

        Returns
        -------
        _: Stat2Dgrid
            Dealiasing grid with size = 3/2 original grid size
        '''
        assert np.mod(self.Nx,4)==0, 'Error, Nx is not a multiple of 4 so dealiasing grid is uneven'
        assert np.mod(self.Ny,4)==0, 'Error, Ny is not a multiple of 4 so dealiasing grid is uneven'
            #Make sure dealising grid has even number of grid point,
        return Stat2Dgrid(self.Lx,int(3*self.Nx/2),self.Ly,int(3*self.Ny/2))
    
    @property
    def Lx(self):
        '''Length of the numerical domain in dimension 0'''
        return self.__Lx
    @property
    def Nx(self):
        '''Number of grid points in dimension 0'''
        return self.__Nx
    @property
    def dx(self):
        '''Grid size in dimension 0'''
        return self.Lx/self.Nx
    @property
    def Ly(self):
        '''Length of the numerical domain in dimension 1'''
        return self.__Ly
    @property
    def Ny(self):
        '''Number of grid points in dimension 1'''
        return self.__Ny
    @property
    def dy(self):
        '''Grid size in dimension 1'''
        return self.Ly/self.Ny
    @property
    def xs(self):
        '''Array with grid points in dimension 0 (real space)'''
        return np.linspace(0,self.Lx,self.Nx,endpoint=False)
    @property
    def ys(self):
        '''Array with grid points in dimension 1 (real space)'''
        return np.linspace(0,self.Ly,self.Ny,endpoint=False)
    @property
    def ks(self):
        '''Array with wave numbers in dimension 0 (Fourier space)'''
        #Use built-in function fftfreq, which works for both even and uneven Nx
        #Also shift wavenumbers so that k=0 is at Nx/2
        return 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(self.Nx,self.dx))
    @property
    def ls(self):
        '''Array with wave numbers in dimension 1 (Fourier space)'''
        #Use built-in function fftfreq, which works for both even and uneven Ny
        #Also shift wavenumbers so that l=0 is at Ny/2
        return 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(self.Ny,self.dy))
    @property
    def Ks(self):
        '''2D mesh with wave numbers in dimension 0 (Fourier space)'''
        Ks, Ls = np.meshgrid(self.ks,self.ls,indexing='ij')
        return Ks
    @property
    def Ls(self):
        '''2D mesh with wave numbers in dimension 1 (Fourier space)'''
        Ks, Ls = np.meshgrid(self.ks,self.ls,indexing='ij')
        return Ls

class Dyn1Dgrid(Stat1Dgrid):
    '''
    Grid for one-dimensional, transient simulation
    '''
    def __init__(self,Lx,Nx,Lt,Nt):
        '''
        Parameters
        ----------
        Lx,Lt: float
            Length of the numerical domain (dimension 0 and time)
        Nx,Nt: int
            Number of grid points (dimension 0 and time)
        '''
        super().__init__(Lx,Nx)
        assert np.mod(Nt,2)==1, 'Error, current implementation only allows even Nt'
            #Don't allow uneven grid sizes,
            #it is implicitly assumed that there are defunct modes

        self.__Lt = Lt
        self.__Nt = Nt
        self.N = self.Nx*self.Nt
        self.shape = (self.Nx,self.Nt)
    
    @property
    def Lt(self):
        '''Time horizon'''
        return self.__Lt
    @property
    def Nt(self):
        '''Number of grid points in time'''
        return self.__Nt
    @property
    def dt(self):
        '''Grid size in time'''
        return self.Lt/self.Nt
    @property
    def ts(self):
        '''Array with grid points in time (real space)'''
        return np.linspace(0,self.Lt,self.Nt,endpoint=False)
    @property
    def omegas(self):
        '''Array with angular frequencies (Fourier space)'''
        #Use built-in function fftfreq, which works for both even and uneven Nt
        #Also shift wavenumbers so that omega=0 is at Nt/2
        return 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(self.Nt,self.dt))
    
class model(object):
    '''
    Common interface for three-layer models
    '''
    def __init__(self,grid,forcing,abl):
        '''
        Parameters
        ----------
        grid: Grid object
            numerical grid
        forcing: CST/WF object (from TLMForcing)
            perturbing force
        abl: ABL object
            atmospheric state
        '''
        self.__grid = grid
        self.__grid32 = self.grid.deal_grid()
        self.__forcing = forcing
        self.__abl = abl
        self.__PHI = None
    
    def solve(self,method='lgmres',verbose=False,WFfeedback=True):
        '''
        Solve the three-layer model

        Parameters
        ----------
        method (optional): str
            method used to solve the linear matrix equation
            default: lgmres
        verbose (optional): bool
            flag for printing solver results
            default: False
        WFfeedback (optional): bool
            flag for including first-order term of wind-farm drag
            default: True
        '''
        N = self.grid.N
        err = 1
        if not WFfeedback:
            #######################
            #Build A and B matrices
            #######################
            if verbose:
                print('Start preprocessing WF model')
                start = time.time()

            self.forcing.preprocess(self.abl,WFfeedback)

            if verbose:
                end = time.time()
                print('WF preprocessing time was',end-start,'s')
                print('Start building matrices')
                start = time.time()

            B = self.Bvector()
            M = self.Moperator()

            if verbose:
                end = time.time()
                print('Time to create matrices was',end-start,'s')
            ##########################
            #Solve system of equations
            ##########################
            if verbose:
                print('Start model calculation')
                start = time.time()

            X = M.matvec(B)

            if verbose:
                end = time.time()
                print('Time to calculate model was',end-start,'s')
            ##########################

        elif method=='direct':
            print('Error, direct methode not supported anymore')
            return 1
            #######################
            #Build A and B matrices
            #######################
            if verbose:
                print('Start building matrices')
            start = time.time()
            A = self.Amatrix()
            B = self.Bvector()
            end = time.time()
            if verbose:
                print('Time to create matrices was',end-start,'s')
            ##########################
            #Solve system of equations
            ##########################
            if verbose:
                print('Start 1D model calculation')
            start = time.time()
            X = scipy.linalg.solve(A,B)
            end = time.time()
            if verbose:
                print('Time to calculate model was',end-start,'s')
            ##########################
        else:
            #######################
            #Build A and B matrices
            #######################
            if verbose:
                print('Start preprocessing WF model')
                start = time.time()

            self.forcing.preprocess(self.abl)

            if verbose:
                end = time.time()
                print('WF preprocessing time was',end-start,'s')
                print('Start building matrices')
                start = time.time()

            B = self.Bvector()
            A = self.Aoperator()
            M = self.Moperator()

            if verbose:
                end = time.time()
                print('Time to create matrices was',end-start,'s')
            ##########################
            #Solve system of equations
            ##########################
            if verbose:
                print('Start model calculation')
                start = time.time()
            if method=='gmres':
                counter = gmres_counter(disp=verbose)
                X,err = scipy.sparse.linalg.gmres(A,B,
                            tol=1.0e-5,
                            restart=100,
                            maxiter=2000,
                            callback=counter)
                print('gmres finished with output flag ',err)
            elif method=='lgmres':
                maxiter = 2000
                counter = lgmres_counter(A,B,maxiter,disp=verbose)
                X,err = scipy.sparse.linalg.lgmres(A,B,
                            tol=1.0e-10,
                            maxiter=maxiter,M=M,
                            callback=counter)
                if verbose:
                    print('lgmres finished with output flag ',err)
                    print('lmgres needed ',counter.niter,' iterations')
#                plt.figure()
#                indices = np.nonzero(counter.residu)[0]
#                plt.semilogy(indices,counter.residu[indices],'-b')
#                plt.show(block=False)
            elif method=='bicgstab':    
                X,err = scipy.sparse.linalg.bicgstab(A,B,maxiter=2000)
                print('bicgstab finished with output flag ',err)
            else:
                print('Method unknown')
                X = np.zeros((B.shape))
            if verbose:
                end = time.time()
                print('Time to calculate model was',end-start,'s')
                rk = A.matvec(X)-B
                print('Euclidean norm of the complex residual is',np.linalg.norm(rk))
            ##########################

        ##################################
        #Store solution and do inverse fft
        ##################################i
        result = self.format_solution(X)
        result['err'] = err
        if verbose:
            print('Start inverse FFT')
            start = time.time()
        result['u1r'], err_u1r   = self.c2r(result['u1c'],True)
        result['v1r'], err_v1r   = self.c2r(result['v1c'],True)
        result['u2r'], err_u2r   = self.c2r(result['u2c'],True)
        result['v2r'], err_v2r   = self.c2r(result['v2c'],True)
        result['etar'], err_etar = self.c2r(result['etac'],True)
        result['pr'], err_pr     = self.c2r(result['pc'],True)
        if verbose:
            end = time.time()
            print('Time to compute inverse FFT was',end-start,'s')
            print('Imaginary part of u1r is smaller than',err_u1r)
            print('Imaginary part of v1r is smaller than',err_v1r)
            print('Imaginary part of u2r is smaller than',err_u2r)
            print('Imaginary part of v2r is smaller than',err_v2r)
            print('Imaginary part of etar is smaller than',err_etar)
            print('Imaginary part of pr is smaller than',err_pr)
        return result

    @property
    def grid(self):
        '''Numerical grid'''
        return self.__grid
    @property
    def grid32(self):
        '''Dealiasing grid'''
        return self.__grid32
    @property
    def abl(self):
        '''Atmospheric state'''
        return self.__abl
    @property
    def forcing(self):
        '''Perturbing force'''
        return self.__forcing
    @property
    def PHI(self):
        '''Complex stratification coefficient'''
        return self.__PHI
    @PHI.setter
    def PHI(self,value):
        self.__PHI = value

class S1Dmodel(model):
    '''
    Steady one-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl,purefriction=False):
        '''
        Parameters
        ----------
        grid: Grid object
            numerical grid
        forcing: CST/WF object (defined in TLMForcing.py)
            perturbing force
        abl: ABL object
            atmospheric state
        purefriction (optional): bool
            flag to consider the pure friction case, i.e., without gravity waves
            default: False
        '''
        super().__init__(grid,forcing,abl)
        if purefriction:
            self.PHI = np.zeros(self.grid.shape,dtype=np.complex128)
        else:
            self.PHI = self.PHIvector()

    def format_solution(self,X):
        '''
        Calculate perturbation quantities from the general solution vector

        Parameters
        ----------
        X: 1d numpy array
            general solution vector

        Returns
        -------
        result: dict
            dictionary with 1d numpy arrays
            keys > u1c,v1c: perturbation velocity in the wind-farm layer
                   u2c,v2c: perturbation velocity in the upper layer
                   etac: inversion displacement
                   pc: pressure perturbation
        '''
        u1c,v1c,u2c,v2c = self.expandX(X)
        etac = self.continuity(u1c,v1c,u2c,v2c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['u2c']  = u2c
        result['v2c']  = v2c
        result['etac'] = etac
        result['pc']   = self.PHI*etac
        return result

    def expandX(self,X):
        '''
        Expand the general solution vector into
        dependent variables of the problem

        Parameters
        ----------
        X: 1d numpy array
            general solution vector

        Returns
        -------
        u1,v1,u2,v2: 1d numpy array
            perturbation velocities in wind-farm and upper layer
        '''
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        u2 = X[2*N:3*N].reshape(self.grid.shape)
        v2 = X[3*N:4*N].reshape(self.grid.shape)
        return u1, v1, u2, v2

    def c2r(self,cfield,returnErr=False):
        '''
        Perform an inverse Fourier transform

        Parameters
        ----------
        cfield: 1d numpy array
            complex field, assuming
                - Hermitian symmetry
                - zero-wavenumber component at the center of the spectrum
        returnErr (optional): bool
            flag to return the maximum imaginary part of the real field
            default: False

        Returns
        -------
        rfield: 1d numpy array
            inverse Fourier transform of cfield (real part)
        err (optional): float
            maximum imaginary part of rfield (zero if input is Hermitian-symmetric)
        '''
        rfield = np.fft.ifft(np.fft.ifftshift(cfield*self.grid.N))
        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def r2c(self,rfield):
        '''
        Perform a Fourier transform

        Parameters
        ----------
        rfield: 1d numpy array
            real field

        Returns
        -------
        cfield: 1d numpy array
            Fourier transform of rfield
            (zero-wavenumber at the center of the spectrum)
        '''
        return np.fft.fftshift(np.fft.fft(rfield))/self.grid.N

    def c2r_deal(self,cfield):
        '''
        Perform an inverse Fourier transform from complex to dealiasing space
        (complex field of size 3/2Nx is obtained by zero-padding)

        Parameters
        ----------
        cfield: 1d numpy array
            complex field, assuming
                - Hermitian symmetry
                - zero-wavenumber component at the center of the spectrum

        Returns
        -------
        rfield32: 1d numpy array
            inverse Fourier transform of cfield32 in dealiasing space (real part)
        '''
        cfield32 = np.concatenate( (
                        np.zeros((int(self.grid.N/4)),dtype=np.complex128),
                        cfield,
                        np.zeros((int(self.grid.N/4)),dtype=np.complex128)) )
        return np.real(np.fft.ifft(np.fft.ifftshift(cfield32*self.grid32.N)))

    def r2c_deal(self,rfield32):
        '''
        Perform a Fourier transform from dealiasing to complex space
        (complex field of size Nx is obtained by disregarding high wavenumbers)

        Parameters
        ----------
        rfield32: 1d numpy array
            real field in dealiasing space

        Returns
        -------
        cfield: 1d numpy array
            Fourier transform of rfield32, disregarding high wavenumbers
            (zero-wavenumber component at the center of the spectrum)
        '''
        cfield32 = np.fft.fftshift(np.fft.fft(rfield32))/self.grid32.N
        return cfield32[int(self.grid.N/4):int(self.grid.N*5/4)]

    def continuity(self,u1c,v1c,u2c,v2c):
        '''Return boundar-layer displacement based on given velocity field'''
        return -self.abl.H1/self.abl.U1*u1c-self.abl.H2/self.abl.U2*u2c

    def Bvector(self):
        #Compute 0th order forcing term (real)
        F0u, F0v = self.forcing.F0(self.abl,self.grid)
        #Convert to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Bu = self.r2c(F0u)/self.abl.H1
        Bv = self.r2c(F0v)/self.abl.H1
        #Defunct modes
        Bu[0] = 0.
        Bv[0] = 0.
        return np.concatenate((Bu,Bv,np.zeros((2*self.grid.Nx,),dtype=np.complex128)))

    def PHIvector(self):
        Nx = self.grid.Nx
        PHIs = self.abl.gprime*np.ones((Nx),dtype=np.complex128)
#        k_lwave = np.sqrt(self.abl.N**2/self.abl.U3**2
#                          +self.abl.gprime**2/self.abl.U3**4*(1-self.abl.Fr**2)**2)
        for index, k in enumerate(self.grid.ks):
            sigma3 = self.abl.U3*k
            #Non-hydrostatic solution 
            if (not sigma3==0): # and (not np.isclose(abs(k),k_lwave,atol=1.0e-4)):
                if sigma3**2>self.abl.N**2:
                    m = 1j*np.sqrt(k**2*np.abs(self.abl.N**2/sigma3**2-1))
                else:
                    m = np.sign(sigma3)*np.sqrt(k**2*(self.abl.N**2/sigma3**2-1))
                PHIs[index] += 1j/m*(self.abl.N**2-sigma3**2)
#            #Hydrostatic solution 
#            if (not sigma3==0):
#                m = np.sign(sigma3)*np.sqrt(k**2*self.abl.N**2/sigma3**2)
#                PHIs[index] += 1j/m*self.abl.N**2
        return PHIs

    def Aoperator(self):
        A = scipy.sparse.linalg.LinearOperator((4*self.grid.N,4*self.grid.N),
                                               matvec=self.Ax,
                                               dtype=np.complex128)
        return A
    
    def Moperator(self):
        M = scipy.sparse.linalg.LinearOperator((4*self.grid.N,4*self.grid.N),
                                               matvec=self.Mx,
                                               dtype=np.complex128)
        return M

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)
        The matrix vector product Mx is found by solving Py=x
        '''
        N = self.grid.N
        M = np.zeros((4,N),dtype=np.complex128)
        X = x.reshape(4,N)
        for index, k in enumerate(self.grid.ks):    
            P = np.zeros((4,4),dtype=np.complex128)
            sigma1 = self.abl.U1*k
            sigma2 = self.abl.U2*k
            #u1 equation
            P[0,0] = (-1j*sigma1
                      +1j*k*self.PHI[index]*self.abl.H1/self.abl.U1
                      -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,1] = (+self.abl.fc
                      -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,2] = (+1j*k*self.PHI[index]*self.abl.H2/self.abl.U2
                      +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,3] =  +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) )
            #v1 equation
            P[1,0] = (-self.abl.fc
                      -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[1,1] = (-1j*sigma1
                      -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[1,2] =  +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) )
            P[1,3] =  +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H1) )
            #u2 equation
            P[2,0] = (+1j*k*self.PHI[index]*self.abl.H1/self.abl.U1
                      +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[2,1] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                    (self.abl.dS12*self.abl.H2) )
            P[2,2] = (-1j*sigma2
                      +1j*k*self.PHI[index]*self.abl.H2/self.abl.U2
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[2,3] = (+self.abl.fc
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H2) ) )
            #v2 equation
            P[3,0] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                    (self.abl.dS12*self.abl.H2) )
            P[3,1] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                    (self.abl.dS12*self.abl.H2) )
            P[3,2] = (-self.abl.fc
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[3,3] = (-1j*sigma2
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        M[:,0] = 0.
        return M.reshape(4*N)

    def Ax(self,x):
        N = self.grid.N
        Axu1 = np.zeros((N),dtype=np.complex128)
        Axv1 = np.zeros((N),dtype=np.complex128)
        Axu2 = np.zeros((N),dtype=np.complex128)
        Axv2 = np.zeros((N),dtype=np.complex128)
        u1 = x[0:N]
        v1 = x[N:2*N]
        u2 = x[2*N:3*N]
        v2 = x[3*N:4*N]
        sigma1 = self.abl.U1*self.grid.ks
        sigma2 = self.abl.U2*self.grid.ks
        #Regular entries
        #u1 equation
        Axu1 += -1j*sigma1*u1
        Axu1 += +1j*self.grid.ks*self.PHI*(self.abl.H1/self.abl.U1*u1+self.abl.H2/self.abl.U2*u2)
        Axu1 += +self.abl.fc*v1
        Axu1 += -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axu1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axu1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        Axu1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        #v1 equation
        Axv1 += -1j*sigma1*v1
        Axv1 += -self.abl.fc*u1
        Axv1 += -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axv1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axv1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        Axv1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        #u2 equation
        Axu2 += -1j*sigma2*u2
        Axu2 += +1j*self.grid.ks*self.PHI*(self.abl.H1/self.abl.U1*u1+self.abl.H2/self.abl.U2*u2)
        Axu2 += +self.abl.fc*v2
        Axu2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)
        Axu2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        #v2 equation
        Axv2 += -1j*sigma2*v2
        Axv2 += -self.abl.fc*u2
        Axv2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        Axv2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)

        #Wind farm forcing: 1st order term
        u1r = self.c2r_deal(u1)
        v1r = self.c2r_deal(v1)
        F1u, F1v = self.forcing.F1(self.abl,self.grid32,u1r,v1r)
        #Convert back to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Axu1 += -self.r2c_deal(F1u)/self.abl.H1
        Axv1 += -self.r2c_deal(F1v)/self.abl.H1

        #Fringe region forcing?
        if self.forcing.fringe:
            u2r = self.c2r_deal(u2)
            v2r = self.c2r_deal(v2)
            F1u1,F1v1,F1u2,F1v2 = self.forcing.F1fringe(self.grid32,
                                                        u1r,v1r,u2r,v2r)
            Axu1 += -self.r2c_deal(F1u1)
            Axv1 += -self.r2c_deal(F1v1)
            Axu2 += -self.r2c_deal(F1u2)
            Axv2 += -self.r2c_deal(F1v2)
    
        #Set defunct modes to zero
        Axu1[0] = 0.
        Axv1[0] = 0.
        Axu2[0] = 0.
        Axv2[0] = 0.
        return np.concatenate((Axu1,Axv1,Axu2,Axv2))

    #Outdated routine
    def Amatrix(self):
        Nx = self.grid.Nx
        A = np.zeros((4*Nx,4*Nx),dtype=np.complex128)
        PHIs = self.PHI
        for index, k in enumerate(self.grid.ks):
            sigma1 = self.abl.U1*k
            sigma2 = self.abl.U2*k
            sigma3 = self.abl.U3*k
            PHI = PHIs[index]
    
            #A matrix regular entries
            #u1 equation
            A[index,index] += -1j*sigma1
            A[index,index] +=  1j*k*PHI*self.abl.H1/self.abl.U1
            A[index,index] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
            A[index,index] += -2*self.abl.C0*self.abl.S1/self.abl.H1
            A[index,index+1*Nx] += self.abl.fc
            A[index,index+2*Nx] += 1j*k*PHI*self.abl.H2/self.abl.U2
            A[index,index+2*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
    
            #v1 equation
            A[index+Nx,index] += -self.abl.fc
            A[index+Nx,index+1*Nx] += -1j*sigma1
            A[index+Nx,index+1*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
            A[index+Nx,index+1*Nx] += -2*self.abl.C0*self.abl.S1/self.abl.H1
            A[index+Nx,index+3*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
    
            #u2 equation
            A[index+2*Nx,index] += 1j*k*PHI*self.abl.H1/self.abl.U1
            A[index+2*Nx,index] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+2*Nx,index+2*Nx] += -1j*sigma2
            A[index+2*Nx,index+2*Nx] +=  1j*k*PHI*self.abl.H2/self.abl.U2
            A[index+2*Nx,index+2*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+2*Nx,index+3*Nx] += self.abl.fc
    
            #v2 equation
            A[index+3*Nx,index+1*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+3*Nx,index+2*Nx] += -self.abl.fc
            A[index+3*Nx,index+3*Nx] += -1j*sigma2
            A[index+3*Nx,index+3*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
        
        #A matrix convolution - Part 1: Linearisation Wind farm drag
        #It is implicitly assumed that a constant forcing model
        #with a CTr field is used. 
        A[:Nx,:Nx] += -2*self.abl.S1/self.abl.H1*cconv_1D(self.r2c(self.forcing.CTr))
        A[Nx:2*Nx,Nx:2*Nx] += -2*self.abl.S1/self.abl.H1*cconv_1D(self.r2c(self.forcing.CTr))

        #A matrix convolution - Part 2: Fringe region forcing
        if self.forcing.Lfringe:
            A[:Nx,:Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[1*Nx:2*Nx,1*Nx:2*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[2*Nx:3*Nx,2*Nx:3*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[3*Nx:4*Nx,3*Nx:4*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
    
        #Set defunct modes to zero
        A[0,:]    = 0.
        A[Nx,:]   = 0.
        A[2*Nx,:] = 0.
        A[3*Nx,:] = 0.
        A[0,0]       = 1.
        A[Nx,Nx]     = 1.
        A[2*Nx,2*Nx] = 1.
        A[3*Nx,3*Nx] = 1.
        return A

class S1Dmodel_old(model):
#Depreciated
#Non-rigorous linearisation of stresses
    '''
    Steady one-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl):
        super().__init__(grid,forcing,abl)
        self.PHI = self.PHIvector()

    def format_solution(self,X):
        u1c,v1c,u2c,v2c = self.expandX(X)
        etac = self.continuity(u1c,v1c,u2c,v2c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['u2c']  = u2c
        result['v2c']  = v2c
        result['etac'] = etac
        result['pc']   = self.PHI*etac
        return result

    def expandX(self,X):
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        u2 = X[2*N:3*N].reshape(self.grid.shape)
        v2 = X[3*N:4*N].reshape(self.grid.shape)
        return u1, v1, u2, v2

    def c2r(self,cfield,returnErr=False):
        rfield = np.fft.ifft(np.fft.ifftshift(cfield*self.grid.N))
        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def r2c(self,rfield):
        return np.fft.fftshift(np.fft.fft(rfield))/self.grid.N

    def continuity(self,u1c,v1c,u2c,v2c):
        '''Return boundar-layer displacement based on given velocity field'''
        return -self.abl.H1/self.abl.U1*u1c-self.abl.H2/self.abl.U2*u2c

    def Bvector(self):
        #Compute 0th order forcing term (real)
        F0u, F0v = self.forcing.F0(self.abl)
        #Convert to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Bu = self.r2c(F0u)/self.abl.H1
        Bv = self.r2c(F0v)/self.abl.H1
        #Defunct modes
        Bu[0] = 0.
        Bv[0] = 0.
        return np.concatenate((Bu,Bv,np.zeros((2*self.grid.Nx,),dtype=np.complex128)))

    def PHIvector(self):
        Nx = self.grid.Nx
        PHIs = self.abl.gprime*np.ones((Nx),dtype=np.complex128)
        for index, k in enumerate(self.grid.ks):
            sigma3 = self.abl.U3*k
            #Non-hydrostatic solution 
            if not sigma3==0:
                if sigma3**2>self.abl.N**2:
                    m = 1j*np.sqrt(k**2*np.abs(self.abl.N**2/sigma3**2-1))
                else:
                    m = np.sign(sigma3)*np.sqrt(k**2*(self.abl.N**2/sigma3**2-1))
                PHIs[index] += 1j/m*(self.abl.N**2-sigma3**2)
        return PHIs

    def Aoperator(self):
        A = scipy.sparse.linalg.LinearOperator((4*self.grid.N,4*self.grid.N),
                                               matvec=self.Ax,
                                               dtype=np.complex128)
        return A
    
    def Moperator(self):
        M = scipy.sparse.linalg.LinearOperator((4*self.grid.N,4*self.grid.N),
                                               matvec=self.Mx,
                                               dtype=np.complex128)
        return M

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)
        The matrix vector product Mx is found by solving Py=x
        '''
        N = self.grid.N
        M = np.zeros((4,N),dtype=np.complex128)
        X = x.reshape(4,N)
        for index, k in enumerate(self.grid.ks):    
            P = np.zeros((4,4),dtype=np.complex128)
            sigma1 = self.abl.U1*k
            sigma2 = self.abl.U2*k
            #u1 equation
            P[0,0] = (-1j*sigma1
                    +1j*k*self.PHI[index]*self.abl.H1/self.abl.U1
                    -2*self.abl.C1*self.abl.dS12/self.abl.H1
                    -2*self.abl.C0*self.abl.S1/self.abl.H1)
            P[0,1] = self.abl.fc
            P[0,2] = (+1j*k*self.PHI[index]*self.abl.H2/self.abl.U2
                    +2*self.abl.C1*self.abl.dS12/self.abl.H1)
            #v1 equation
            P[1,0] = -self.abl.fc
            P[1,1] = (-1j*sigma1
                    -2*self.abl.C1*self.abl.dS12/self.abl.H1
                    -2*self.abl.C0*self.abl.S1/self.abl.H1)
            P[1,3] = 2*self.abl.C1*self.abl.dS12/self.abl.H1
            #u2 equation
            P[2,0] = (+1j*k*self.PHI[index]*self.abl.H1/self.abl.U1
                     +2*self.abl.C1*self.abl.dS12/self.abl.H2)
            P[2,2] = (-1j*sigma2
                    +1j*k*self.PHI[index]*self.abl.H2/self.abl.U2
                    -2*self.abl.C1*self.abl.dS12/self.abl.H2)
            P[2,3] = self.abl.fc
            #v2 equation
            P[3,1] = 2*self.abl.C1*self.abl.dS12/self.abl.H2
            P[3,2] = -self.abl.fc
            P[3,3] = (-1j*sigma2
                    -2*self.abl.C1*self.abl.dS12/self.abl.H2)
            M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        M[:,0] = 0.
        return M.reshape(4*N)

    def Ax(self,x):
        N = self.grid.N
        Axu1 = np.zeros((N),dtype=np.complex128)
        Axv1 = np.zeros((N),dtype=np.complex128)
        Axu2 = np.zeros((N),dtype=np.complex128)
        Axv2 = np.zeros((N),dtype=np.complex128)
        u1 = x[0:N]
        v1 = x[N:2*N]
        u2 = x[2*N:3*N]
        v2 = x[3*N:4*N]
        sigma1 = self.abl.U1*self.grid.ks
        sigma2 = self.abl.U2*self.grid.ks
        #Regular entries
        #u1 equation
        Axu1 += (-1j*sigma1
                +1j*self.grid.ks*self.PHI*self.abl.H1/self.abl.U1
                -2*self.abl.C1*self.abl.dS12/self.abl.H1
                -2*self.abl.C0*self.abl.S1/self.abl.H1)*u1
        Axu1 += self.abl.fc*v1
        Axu1 += (+1j*self.grid.ks*self.PHI*self.abl.H2/self.abl.U2
                +2*self.abl.C1*self.abl.dS12/self.abl.H1)*u2
        #v1 equation
        Axv1 += -self.abl.fc*u1
        Axv1 += (-1j*sigma1
                -2*self.abl.C1*self.abl.dS12/self.abl.H1
                -2*self.abl.C0*self.abl.S1/self.abl.H1)*v1
        Axv1 += 2*self.abl.C1*self.abl.dS12/self.abl.H1*v2
        #u2 equation
        Axu2 += (+1j*self.grid.ks*self.PHI*self.abl.H1/self.abl.U1
                 +2*self.abl.C1*self.abl.dS12/self.abl.H2)*u1
        Axu2 += (-1j*sigma2
                +1j*self.grid.ks*self.PHI*self.abl.H2/self.abl.U2
                -2*self.abl.C1*self.abl.dS12/self.abl.H2)*u2
        Axu2 += self.abl.fc*v2
        #v2 equation
        Axv2 += 2*self.abl.C1*self.abl.dS12/self.abl.H2*v1
        Axv2 += -self.abl.fc*u2
        Axv2 += (-1j*sigma2
                -2*self.abl.C1*self.abl.dS12/self.abl.H2)*v2

        #Wind farm forcing: 1st order term
        u1r = self.c2r(u1)
        v1r = self.c2r(v1)
        F1u, F1v = self.forcing.F1(self.abl,u1r,v1r)
        #Convert back to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Axu1 += -self.r2c(F1u)/self.abl.H1
        Axv1 += -self.r2c(F1v)/self.abl.H1

        #Fringe region forcing?
        if self.forcing.fringe:
            u2r = self.c2r(u2)
            v2r = self.c2r(v2)
            F1u1,F1v1,F1u2,F1v2 = self.forcing.F1fringe(u1r,v1r,u2r,v2r)
            Axu1 += -self.r2c(F1u1)
            Axv1 += -self.r2c(F1v1)
            Axu2 += -self.r2c(F1u2)
            Axv2 += -self.r2c(F1v2)
    
        #Set defunct modes to zero
        Axu1[0] = 0.
        Axv1[0] = 0.
        Axu2[0] = 0.
        Axv2[0] = 0.
        return np.concatenate((Axu1,Axv1,Axu2,Axv2))

    #Outdated routine
    def Amatrix(self):
        Nx = self.grid.Nx
        A = np.zeros((4*Nx,4*Nx),dtype=np.complex128)
        PHIs = self.PHI
        for index, k in enumerate(self.grid.ks):
            sigma1 = self.abl.U1*k
            sigma2 = self.abl.U2*k
            sigma3 = self.abl.U3*k
            PHI = PHIs[index]
    
            #A matrix regular entries
            #u1 equation
            A[index,index] += -1j*sigma1
            A[index,index] +=  1j*k*PHI*self.abl.H1/self.abl.U1
            A[index,index] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
            A[index,index] += -2*self.abl.C0*self.abl.S1/self.abl.H1
            A[index,index+1*Nx] += self.abl.fc
            A[index,index+2*Nx] += 1j*k*PHI*self.abl.H2/self.abl.U2
            A[index,index+2*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
    
            #v1 equation
            A[index+Nx,index] += -self.abl.fc
            A[index+Nx,index+1*Nx] += -1j*sigma1
            A[index+Nx,index+1*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
            A[index+Nx,index+1*Nx] += -2*self.abl.C0*self.abl.S1/self.abl.H1
            A[index+Nx,index+3*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
    
            #u2 equation
            A[index+2*Nx,index] += 1j*k*PHI*self.abl.H1/self.abl.U1
            A[index+2*Nx,index] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+2*Nx,index+2*Nx] += -1j*sigma2
            A[index+2*Nx,index+2*Nx] +=  1j*k*PHI*self.abl.H2/self.abl.U2
            A[index+2*Nx,index+2*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+2*Nx,index+3*Nx] += self.abl.fc
    
            #v2 equation
            A[index+3*Nx,index+1*Nx] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
            A[index+3*Nx,index+2*Nx] += -self.abl.fc
            A[index+3*Nx,index+3*Nx] += -1j*sigma2
            A[index+3*Nx,index+3*Nx] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
        
        #A matrix convolution - Part 1: Linearisation Wind farm drag
        #It is implicitly assumed that a constant forcing model
        #with a CTr field is used. 
        A[:Nx,:Nx] += -2*self.abl.S1/self.abl.H1*cconv_1D(self.r2c(self.forcing.CTr))
        A[Nx:2*Nx,Nx:2*Nx] += -2*self.abl.S1/self.abl.H1*cconv_1D(self.r2c(self.forcing.CTr))

        #A matrix convolution - Part 2: Fringe region forcing
        if self.forcing.Lfringe:
            A[:Nx,:Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[1*Nx:2*Nx,1*Nx:2*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[2*Nx:3*Nx,2*Nx:3*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
            A[3*Nx:4*Nx,3*Nx:4*Nx] += -cconv_1D(self.r2c(self.forcing.Cfr))
    
        #Set defunct modes to zero
        A[0,:]    = 0.
        A[Nx,:]   = 0.
        A[2*Nx,:] = 0.
        A[3*Nx,:] = 0.
        A[0,0]       = 1.
        A[Nx,Nx]     = 1.
        A[2*Nx,2*Nx] = 1.
        A[3*Nx,3*Nx] = 1.
        return A

class S1DPmodel(S1Dmodel):
    '''
    Steady one-dimensional pressure model
    (Three-layer model, but p is an input)
    '''
    def __init__(self,grid,forcing,abl,pressure):
        super().__init__(grid,forcing,abl)
        if pressure.shape==self.grid.shape:
            self.__pc = self.r2c(pressure)
        else:
            print('Error: input data for pressure does not have the correct shape')
            return

    def format_solution(self,X):
        u1c,v1c,u2c,v2c = self.expandX(X)
        etac = self.continuity(u1c,v1c,u2c,v2c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['u2c']  = u2c
        result['v2c']  = v2c
        result['etac'] = etac
        result['pc']   = self.pc
        return result

    def Bvector(self):
        #Compute 0th order forcing term (real)
        F0u, F0v = self.forcing.F0(self.abl,self.grid)
        #Convert to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Bu1 = self.r2c(F0u)/self.abl.H1 + 1j*self.grid.ks*self.pc
        Bv1 = self.r2c(F0v)/self.abl.H1
        Bu2 = 1j*self.grid.ks*self.pc
        Bv2 = np.zeros((self.grid.Nx),dtype=np.complex128)
        #Defunct modes
        Bu1[0] = 0.
        Bv1[0] = 0.
        Bu2[0] = 0.
        Bv2[0] = 0.
        return np.concatenate((Bu1,Bv1,Bu2,Bv2))

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)
        The matrix vector product Mx is found by solving Py=x
        '''
        N = self.grid.N
        M = np.zeros((4,N),dtype=np.complex128)
        X = x.reshape(4,N)
        for index, k in enumerate(self.grid.ks):    
            P = np.zeros((4,4),dtype=np.complex128)
            sigma1 = self.abl.U1*k
            sigma2 = self.abl.U2*k
            #u1 equation
            P[0,0] = (-1j*sigma1
                      -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,1] = (+self.abl.fc
                      -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,2] = (+self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[0,3] =  +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) )
            #v1 equation
            P[1,0] = (-self.abl.fc
                      -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[1,1] = (-1j*sigma1
                      -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                                     (self.abl.S1*self.abl.H1) )
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H1) ) )
            P[1,2] =  +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H1) )
            P[1,3] =  +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H1) )
            #u2 equation
            P[2,0] = (self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[2,1] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                    (self.abl.dS12*self.abl.H2) )
            P[2,2] = (-1j*sigma2
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[2,3] = (+self.abl.fc
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H2) ) )
            #v2 equation
            P[3,0] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                    (self.abl.dS12*self.abl.H2) )
            P[3,1] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                    (self.abl.dS12*self.abl.H2) )
            P[3,2] = (-self.abl.fc
                      -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                     (self.abl.dS12*self.abl.H2) ) )
            P[3,3] = (-1j*sigma2
                      -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                     (self.abl.dS12*self.abl.H2) ) )
            M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        M[:,0] = 0.
        return M.reshape(4*N)

    def Ax(self,x):
        N = self.grid.N
        Axu1 = np.zeros((N),dtype=np.complex128)
        Axv1 = np.zeros((N),dtype=np.complex128)
        Axu2 = np.zeros((N),dtype=np.complex128)
        Axv2 = np.zeros((N),dtype=np.complex128)
        u1 = x[0:N]
        v1 = x[N:2*N]
        u2 = x[2*N:3*N]
        v2 = x[3*N:4*N]
        sigma1 = self.abl.U1*self.grid.ks
        sigma2 = self.abl.U2*self.grid.ks
        #Regular entries
        #u1 equation
        Axu1 += -1j*sigma1*u1
        Axu1 += +self.abl.fc*v1
        Axu1 += -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axu1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axu1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        Axu1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        #v1 equation
        Axv1 += -1j*sigma1*v1
        Axv1 += -self.abl.fc*u1
        Axv1 += -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axv1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axv1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        Axv1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        #u2 equation
        Axu2 += -1j*sigma2*u2
        Axu2 += +self.abl.fc*v2
        Axu2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)
        Axu2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        #v2 equation
        Axv2 += -1j*sigma2*v2
        Axv2 += -self.abl.fc*u2
        Axv2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        Axv2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)

        #Wind farm forcing: 1st order term
        u1r = self.c2r_deal(u1)
        v1r = self.c2r_deal(v1)
        F1u, F1v = self.forcing.F1(self.abl,self.grid32,u1r,v1r)
        #Convert back to fourier space and
        #divide by H1 (TLM solves height-averaged equations)
        Axu1 += -self.r2c_deal(F1u)/self.abl.H1
        Axv1 += -self.r2c_deal(F1v)/self.abl.H1

        #Fringe region forcing?
        if self.forcing.fringe:
            u2r = self.c2r_deal(u2)
            v2r = self.c2r_deal(v2)
            F1u1,F1v1,F1u2,F1v2 = self.forcing.F1fringe(self.grid32,
                                                        u1r,v1r,u2r,v2r)
            Axu1 += -self.r2c_deal(F1u1)
            Axv1 += -self.r2c_deal(F1v1)
            Axu2 += -self.r2c_deal(F1u2)
            Axv2 += -self.r2c_deal(F1v2)
    
        #Set defunct modes to zero
        Axu1[0] = 0.
        Axv1[0] = 0.
        Axu2[0] = 0.
        Axv2[0] = 0.
        return np.concatenate((Axu1,Axv1,Axu2,Axv2))

    @property
    def pc(self):
        return self.__pc

class S2Dmodel(model):
    '''
    Steady two-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl,purefriction=False):
        super().__init__(grid,forcing,abl)
        if purefriction:
            self.PHI = np.zeros(self.grid.shape,dtype=np.complex128)
        else:
            self.PHI = self.PHIvector()

    def format_solution(self,X):
        u1c,v1c,u2c,v2c,p1c,p2c = self.expandX(X)
        pc = p1c+p2c
        etac = self.continuity(u1c,v1c,u2c,v2c,p1c,p2c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['u2c']  = u2c
        result['v2c']  = v2c
        result['etac'] = etac
        result['pc']   = pc
        result['p1c']  = p1c
        result['p2c']  = p2c
        return result

    def expandX(self,X):
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        u2 = X[2*N:3*N].reshape(self.grid.shape)
        v2 = X[3*N:4*N].reshape(self.grid.shape)
        p1 = X[4*N:5*N].reshape(self.grid.shape)
        p2 = X[5*N:6*N].reshape(self.grid.shape)
        return u1, v1, u2, v2, p1, p2

    def c2r(self,cfield,returnErr=False):
        rfield = np.fft.ifft2(np.fft.ifftshift(cfield*self.grid.N))
        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def r2c(self,rfield):
        return np.fft.fftshift(np.fft.fft2(rfield))/self.grid.N

    def c2r_deal(self,cfield):
        #First padd in x-direction
        cfield32 = np.concatenate( (
            np.zeros((int(self.grid.Nx/4),self.grid.Ny),dtype=np.complex128),
            cfield,
            np.zeros((int(self.grid.Nx/4),self.grid.Ny),dtype=np.complex128) ),
                                   axis=0 )
        #Then padd in y-direction
        cfield32 = np.concatenate( (
            np.zeros((self.grid32.Nx,int(self.grid.Ny/4)),dtype=np.complex128),
            cfield32,
            np.zeros((self.grid32.Nx,int(self.grid.Ny/4)),dtype=np.complex128) ),
                                   axis=1)

        rfield = np.fft.ifft2(np.fft.ifftshift(cfield32*self.grid32.N))

        return np.real(rfield)

    def r2c_deal(self,rfield32):
        cfield32 = np.fft.fftshift(np.fft.fft2(rfield32))/self.grid32.N
        return cfield32[int(self.grid.Nx/4):int(self.grid.Nx*5/4),int(self.grid.Ny/4):int(self.grid.Ny*5/4)]

    def continuity(self,u1c,v1c,u2c,v2c,p1c,p2c):
        '''Return boundar-layer displacement based on given velocity field'''
        ##Using the continuity equation
        #Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        #sigma1 = self.abl.U1*Ks+self.abl.V1*Ls
        #sigma2 = self.abl.U2*Ks+self.abl.V2*Ls
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    eta1 = -self.abl.H1/sigma1*(Ks*u1c+Ls*v1c)
        #    eta2 = -self.abl.H2/sigma2*(Ks*u2c+Ls*v2c)
        ##When sigma1,2 is exactly zero (this includes the mean mode),
        ##eta1,2 is undefined (division by zero). The exact value doesn't matter
        ##because it does not appear in the system of equations (the continuity
        ##equation changes to an incompressiblity condition in the limiting case)
        ##The value of eta1,2 is replaced with the limit value p1,2/PHI to make
        ##the eta field continuous. Note that setting eta1,2 to zero would cause
        ##broad stripes in the solution field when one of U1,V1,U2,V2 is zero.
        ##There is no issue for values of sigma1,2 close but not equal to zero
        #eta1[sigma1==0.]=p1c[sigma1==0.]/self.PHI[sigma1==0.]
        #eta2[sigma2==0.]=p2c[sigma2==0.]/self.PHI[sigma2==0.]
        #eta = eta1+eta2

        #Using the pressure field:
        #Eta can also be found by the relation p1,2=PHI*eta1,2
        #The result is identical to the displacement found with the continuity
        #equation when the limiting values for sigma1,2->0 are chosen correctly
        #However, this method is numerically more stable as it only involves a
        #product, whereas the continuity approach can result in the division of
        #two very small numbers (order of 1.0e-21) which is inaccurate
        with np.errstate(divide='ignore',invalid='ignore'):
            eta1 = p1c/self.PHI
            eta2 = p2c/self.PHI
        #The only issue here is when PHI=0., which can occur for gprime=0.
        Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        sigma1 = self.abl.U1*Ks+self.abl.V1*Ls
        sigma2 = self.abl.U2*Ks+self.abl.V2*Ls
        with np.errstate(divide='ignore',invalid='ignore'):
            eta1[self.PHI==0.] = -self.abl.H1*(Ks[self.PHI==0.]*u1c[self.PHI==0.]+Ls[self.PHI==0.]*v1c[self.PHI==0.])/sigma1[self.PHI==0.]
            eta2[self.PHI==0.] = -self.abl.H2*(Ks[self.PHI==0.]*u2c[self.PHI==0.]+Ls[self.PHI==0.]*v2c[self.PHI==0.])/sigma2[self.PHI==0.]
        #Careful now when sigma1,2 is close to zero
        #in fact, atol=1.e-8 still doesn't provide satifactory results
        eta1[(self.PHI==0.) & (np.isclose(sigma1,0.,atol=1.e-8))] = 0.
        eta2[(self.PHI==0.) & (np.isclose(sigma2,0.,atol=1.e-8))] = 0.
        eta = eta1 + eta2
        return eta

    def Bvector(self):
        #Compute 0th order forcing term (2D real)
        F0u, F0v = self.forcing.F0(self.abl,self.grid)
        #Convert to fourier space, cast into 1D array and
        #divide by H1 (TLM solves height-averaged equations)
        Bu = np.ravel(self.r2c(F0u))/self.abl.H1
        Bv = np.ravel(self.r2c(F0v))/self.abl.H1
        #Set defunct modes to zero
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        Bu[defunctindices] = 0.
        Bv[defunctindices] = 0.
        return np.concatenate((Bu,Bv,np.zeros((4*Nx*Ny,),dtype=np.complex128)))

    def PHIvector(self):
        PHI = self.abl.gprime*np.ones(self.grid.shape,dtype=np.complex128)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                sigma3 = self.abl.U3*k+self.abl.V3*l
                #Non-hydrostatic solution 
                if not sigma3==0:
                    if sigma3**2>self.abl.N**2:
                        m = 1j*np.sqrt((k**2+l**2)*np.abs(self.abl.N**2/sigma3**2-1))
                    else:
                        m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2-1))
                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2-sigma3**2)
#                #Hydrostatic solution 
#                if not sigma3==0:
#                    m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2))
#                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2)
        return PHI

    def Aoperator(self):
        A = scipy.sparse.linalg.LinearOperator((6*self.grid.N,6*self.grid.N),
                                               matvec=self.Ax,
                                               dtype=np.complex128)
        return A
    
    def Moperator(self):
        M = scipy.sparse.linalg.LinearOperator((6*self.grid.N,6*self.grid.N),
                                               matvec=self.Mx,
                                               dtype=np.complex128)
        return M

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)
        The matrix vector product Mx is found by solving Py=x
        '''
        N = self.grid.N
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        M = np.zeros((6,N),dtype=np.complex128)
        X = x.reshape(6,N)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                index = indexl + Ny*indexk
                P = np.zeros((6,6),dtype=np.complex128)
                sigma1 = self.abl.U1*k+self.abl.V1*l
                sigma2 = self.abl.U2*k+self.abl.V2*l
                #u1 equation
                P[0,0] = (-1j*sigma1
                          -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                                         (self.abl.S1*self.abl.H1) )
                          -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                         (self.abl.dS12*self.abl.H1) ) 
                          -self.abl.nu1*(k**2+l**2) )
                P[0,1] = (+self.abl.fc
                          -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                         (self.abl.S1*self.abl.H1) )
                          -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                         (self.abl.dS12*self.abl.H1) ) )
                P[0,2] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                        (self.abl.dS12*self.abl.H1) )
                P[0,3] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                        (self.abl.dS12*self.abl.H1) )
                P[0,4] = -1j*k
                P[0,5] = -1j*k
                #v1 equation
                P[1,0] = (-self.abl.fc
                          -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                                         (self.abl.S1*self.abl.H1) )
                          -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                         (self.abl.dS12*self.abl.H1) ) )
                P[1,1] = (-1j*sigma1
                          -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                                         (self.abl.S1*self.abl.H1) )
                          -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                         (self.abl.dS12*self.abl.H1) )
                          -self.abl.nu1*(k**2+l**2) )
                P[1,2] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                        (self.abl.dS12*self.abl.H1) )
                P[1,3] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                        (self.abl.dS12*self.abl.H1) )
                P[1,4] = -1j*l
                P[1,5] = -1j*l
                #u2 equation
                P[2,0] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                        (self.abl.dS12*self.abl.H2) )
                P[2,1] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                        (self.abl.dS12*self.abl.H2) )
                P[2,2] = (-1j*sigma2
                          -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                                         (self.abl.dS12*self.abl.H2) )
                          -self.abl.nu2*(k**2+l**2) )
                P[2,3] = (+self.abl.fc
                          -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                         (self.abl.dS12*self.abl.H2) ) )
                P[2,4] = -1j*k
                P[2,5] = -1j*k
                #v2 equation
                P[3,0] = +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                        (self.abl.dS12*self.abl.H2) )
                P[3,1] = +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                        (self.abl.dS12*self.abl.H2) )
                P[3,2] = (-self.abl.fc
                          -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                                         (self.abl.dS12*self.abl.H2) ) )
                P[3,3] = (-1j*sigma2
                          -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                                         (self.abl.dS12*self.abl.H2) )
                          -self.abl.nu2*(k**2+l**2) )
                P[3,4] = -1j*l
                P[3,5] = -1j*l
                #p1 equation
                #General case
                P[4,0] = self.PHI[indexk,indexl]*self.abl.H1*k
                P[4,1] = self.PHI[indexk,indexl]*self.abl.H1*l
                P[4,4] = sigma1
                #p2 equation
                #General case
                P[5,2] = self.PHI[indexk,indexl]*self.abl.H2*k
                P[5,3] = self.PHI[indexk,indexl]*self.abl.H2*l
                P[5,5] = sigma2
                #k=l=0
                if k==0 and l == 0:
                    P[4,4] = 1.+0.j
                    P[5,5] = 1.+0.j
                #PHI=0
                if self.PHI[indexk,indexl]==0.:
                    P[4,4] = 1.+0.j
                    P[5,5] = 1.+0.j

                M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        M[:,defunctindices] = 0.+0.j
        return M.reshape(6*N)

    def Ax(self,x):
        N = self.grid.N
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        #Initialise components of the matrix vector product
        Axu1 = np.zeros((N),dtype=np.complex128)
        Axv1 = np.zeros((N),dtype=np.complex128)
        Axu2 = np.zeros((N),dtype=np.complex128)
        Axv2 = np.zeros((N),dtype=np.complex128)
        Axp1 = np.zeros((N),dtype=np.complex128)
        Axp2 = np.zeros((N),dtype=np.complex128)
        #Extract dependent variables (u1,v1,u2,v2,p1,p2) from the vector
        u1 = x[0:N]
        v1 = x[N:2*N]
        u2 = x[2*N:3*N]
        v2 = x[3*N:4*N]
        p1 = x[4*N:5*N]
        p2 = x[5*N:6*N]
        #Create some additional vectors
        Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        ks = np.ravel(Ks)
        ls = np.ravel(Ls)
        sigma1 = self.abl.U1*ks+self.abl.V1*ls
        sigma2 = self.abl.U2*ks+self.abl.V2*ls
        #Regular entries
        #u1 equation
        Axu1 += -1j*sigma1*u1
        Axu1 += -1j*ks*(p1+p2)
        Axu1 += +self.abl.fc*v1
        Axu1 += -self.abl.C0*( (self.abl.S1**2+self.abl.U1**2)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axu1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axu1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        Axu1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        Axu1 += -self.abl.nu1*(ks**2+ls**2)*u1
        #v1 equation
        Axv1 += -1j*sigma1*v1
        Axv1 += -1j*ls*(p1+p2)
        Axv1 += -self.abl.fc*u1
        Axv1 += -self.abl.C0*( (self.abl.S1**2+self.abl.V1**2)/
                    (self.abl.S1*self.abl.H1) )*v1
        Axv1 += -self.abl.C0*( (self.abl.U1*self.abl.V1)/
                    (self.abl.S1*self.abl.H1) )*u1
        Axv1 += -self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H1) )*(v1-v2)
        Axv1 += -self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H1) )*(u1-u2)
        Axv1 += -self.abl.nu1*(ks**2+ls**2)*v1
        #u2 equation
        Axu2 += -1j*sigma2*u2
        Axu2 += -1j*ks*(p1+p2)
        Axu2 += +self.abl.fc*v2
        Axu2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dU12**2)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)
        Axu2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        Axu2 += -self.abl.nu2*(ks**2+ls**2)*u2
        #v2 equation
        Axv2 += -1j*sigma2*v2
        Axv2 += -1j*ls*(p1+p2)
        Axv2 += -self.abl.fc*u2
        Axv2 += +self.abl.C1*( (self.abl.dS12**2+self.abl.dV12**2)/
                    (self.abl.dS12*self.abl.H2) )*(v1-v2)
        Axv2 += +self.abl.C1*( (self.abl.dU12*self.abl.dV12)/
                    (self.abl.dS12*self.abl.H2) )*(u1-u2)
        Axv2 += -self.abl.nu2*(ks**2+ls**2)*v2
        #p1 equation
        #General case
        Axp1 += np.ravel(self.PHI)*self.abl.H1*ks*u1
        Axp1 += np.ravel(self.PHI)*self.abl.H1*ls*v1
        Axp1 += sigma1*p1
        #p2 equation
        #General case
        Axp2 += np.ravel(self.PHI)*self.abl.H2*ks*u2
        Axp2 += np.ravel(self.PHI)*self.abl.H2*ls*v2
        Axp2 += sigma2*p2
        #p1,2 equation: k=l=0
        indices_klzero = int(Nx/2)*Ny+int(Ny/2)
        Axp1[indices_klzero] = p1[indices_klzero]
        Axp2[indices_klzero] = p2[indices_klzero]

        #Compute 1st order forcing term
        u1r = self.c2r_deal(u1.reshape(self.grid.shape))
        v1r = self.c2r_deal(v1.reshape(self.grid.shape))
        F1u, F1v = self.forcing.F1(self.abl,self.grid32,u1r,v1r)
        #Convert back to fourier space, cast into 1D array and
        #divide by H1 (TLM solves height-averaged equations)
        Axu1 += -np.ravel(self.r2c_deal(F1u))/self.abl.H1
        Axv1 += -np.ravel(self.r2c_deal(F1v))/self.abl.H1

        #Set defunct modes to zero
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        Axu1[defunctindices] = 0.
        Axv1[defunctindices] = 0.
        Axu2[defunctindices] = 0.
        Axv2[defunctindices] = 0.
        Axp1[defunctindices] = 0.
        Axp2[defunctindices] = 0.
        return np.concatenate((Axu1,Axv1,Axu2,Axv2,Axp1,Axp2))

class S2Dmodel_old(model):
#depreciated
#Non-rigorous linearisation of stresses
    '''
    Steady two-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl):
        super().__init__(grid,forcing,abl)
        self.PHI = self.PHIvector()

    def format_solution(self,X):
        u1c,v1c,u2c,v2c,p1c,p2c = self.expandX(X)
        pc = p1c+p2c
        etac = self.continuity(u1c,v1c,u2c,v2c,p1c,p2c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['u2c']  = u2c
        result['v2c']  = v2c
        result['etac'] = etac
        result['pc']   = pc
        result['p1c']  = p1c
        result['p2c']  = p2c
        return result

    def expandX(self,X):
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        u2 = X[2*N:3*N].reshape(self.grid.shape)
        v2 = X[3*N:4*N].reshape(self.grid.shape)
        p1 = X[4*N:5*N].reshape(self.grid.shape)
        p2 = X[5*N:6*N].reshape(self.grid.shape)
        return u1, v1, u2, v2, p1, p2

    def c2r(self,cfield,returnErr=False):
        rfield = np.fft.ifft2(np.fft.ifftshift(cfield*self.grid.N))
        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def r2c(self,rfield):
        return np.fft.fftshift(np.fft.fft2(rfield))/self.grid.N

    def continuity(self,u1c,v1c,u2c,v2c,p1c,p2c):
        '''Return boundar-layer displacement based on given velocity field'''
        ##Using the continuity equation
        #Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        #sigma1 = self.abl.U1*Ks+self.abl.V1*Ls
        #sigma2 = self.abl.U2*Ks+self.abl.V2*Ls
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    eta1 = -self.abl.H1/sigma1*(Ks*u1c+Ls*v1c)
        #    eta2 = -self.abl.H2/sigma2*(Ks*u2c+Ls*v2c)
        ##When sigma1,2 is exactly zero (this includes the mean mode),
        ##eta1,2 is undefined (division by zero). The exact value doesn't matter
        ##because it does not appear in the system of equations (the continuity
        ##equation changes to an incompressiblity condition in the limiting case)
        ##The value of eta1,2 is replaced with the limit value p1,2/PHI to make
        ##the eta field continuous. Note that setting eta1,2 to zero would cause
        ##broad stripes in the solution field when one of U1,V1,U2,V2 is zero.
        ##There is no issue for values of sigma1,2 close but not equal to zero
        #eta1[sigma1==0.]=p1c[sigma1==0.]/self.PHI[sigma1==0.]
        #eta2[sigma2==0.]=p2c[sigma2==0.]/self.PHI[sigma2==0.]
        #eta = eta1+eta2

        #Using the pressure field:
        #Eta can also be found by the relation p1,2=PHI*eta1,2
        #The result is identical to the displacement found with the continuity
        #equation when the limiting values for sigma1,2->0 are chosen correctly
        #However, this method is numerically more stable as it only involves a
        #product, whereas the continuity approach can result in the division of
        #two very small numbers (order of 1.0e-21) which is inaccurate
        #The only issue here is when PHI=0., which can occur for gprime=0.
        eta1 = p1c/self.PHI
        eta2 = p2c/self.PHI
        eta = eta1 + eta2
        return eta

    def Bvector(self):
        #Compute 0th order forcing term (2D real)
        F0u, F0v = self.forcing.F0(self.abl)
        #Convert to fourier space, cast into 1D array and
        #divide by H1 (TLM solves height-averaged equations)
        Bu = np.ravel(self.r2c(F0u))/self.abl.H1
        Bv = np.ravel(self.r2c(F0v))/self.abl.H1
        #Set defunct modes to zero
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        Bu[defunctindices] = 0.
        Bv[defunctindices] = 0.
        return np.concatenate((Bu,Bv,np.zeros((4*Nx*Ny,),dtype=np.complex128)))

    def PHIvector(self):
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        PHI = self.abl.gprime*np.ones((Nx,Ny),dtype=np.complex128)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                sigma3 = self.abl.U3*k+self.abl.V3*l
                #Non-hydrostatic solution 
                if not sigma3==0:
                    if sigma3**2>self.abl.N**2:
                        m = 1j*np.sqrt((k**2+l**2)*np.abs(self.abl.N**2/sigma3**2-1))
                    else:
                        m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2-1))
                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2-sigma3**2)
        return PHI

    def Aoperator(self):
        A = scipy.sparse.linalg.LinearOperator((6*self.grid.N,6*self.grid.N),
                                               matvec=self.Ax,
                                               dtype=np.complex128)
        return A
    
    def Moperator(self):
        M = scipy.sparse.linalg.LinearOperator((6*self.grid.N,6*self.grid.N),
                                               matvec=self.Mx,
                                               dtype=np.complex128)
        return M

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (excluding the convolution products)
        The matrix vector product Mx is found by solving Py=x
        '''
        N = self.grid.N
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        M = np.zeros((6,N),dtype=np.complex128)
        X = x.reshape(6,N)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                index = indexl + Ny*indexk
                P = np.zeros((6,6),dtype=np.complex128)
                sigma1 = self.abl.U1*k+self.abl.V1*l
                sigma2 = self.abl.U2*k+self.abl.V2*l
                #u1 equation
                P[0,0] = (-1j*sigma1
                        -2*self.abl.C1*self.abl.dS12/self.abl.H1
                        -2*self.abl.C0*self.abl.S1/self.abl.H1)
                P[0,1] = self.abl.fc
                P[0,2] = 2*self.abl.C1*self.abl.dS12/self.abl.H1
                P[0,4] = -1j*k
                P[0,5] = -1j*k
                #v1 equation
                P[1,0] = -self.abl.fc
                P[1,1] = (-1j*sigma1
                        -2*self.abl.C1*self.abl.dS12/self.abl.H1
                        -2*self.abl.C0*self.abl.S1/self.abl.H1)
                P[1,3] = 2*self.abl.C1*self.abl.dS12/self.abl.H1
                P[1,4] = -1j*l
                P[1,5] = -1j*l
                #u2 equation
                P[2,0] = 2*self.abl.C1*self.abl.dS12/self.abl.H2
                P[2,2] = (-1j*sigma2
                        -2*self.abl.C1*self.abl.dS12/self.abl.H2)
                P[2,3] = self.abl.fc
                P[2,4] = -1j*k
                P[2,5] = -1j*k
                #v2 equation
                P[3,1] = 2*self.abl.C1*self.abl.dS12/self.abl.H2
                P[3,2] = -self.abl.fc
                P[3,3] = (-1j*sigma2
                        -2*self.abl.C1*self.abl.dS12/self.abl.H2)
                P[3,4] = -1j*l
                P[3,5] = -1j*l
                #p1 equation
                #General case
                P[4,0] = self.PHI[indexk,indexl]*self.abl.H1*k
                P[4,1] = self.PHI[indexk,indexl]*self.abl.H1*l
                P[4,4] = sigma1
                #p2 equation
                #General case
                P[5,2] = self.PHI[indexk,indexl]*self.abl.H2*k
                P[5,3] = self.PHI[indexk,indexl]*self.abl.H2*l
                P[5,5] = sigma2
                #k=l=0
                if k==0 and l == 0:
                    P[4,4] = 1.+0.j
                    P[5,5] = 1.+0.j

                M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        M[:,defunctindices] = 0.+0.j
        return M.reshape(6*N)

    def Ax(self,x):
        N = self.grid.N
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        #Initialise components of the matrix vector product
        Axu1 = np.zeros((N),dtype=np.complex128)
        Axv1 = np.zeros((N),dtype=np.complex128)
        Axu2 = np.zeros((N),dtype=np.complex128)
        Axv2 = np.zeros((N),dtype=np.complex128)
        Axp1 = np.zeros((N),dtype=np.complex128)
        Axp2 = np.zeros((N),dtype=np.complex128)
        #Extract dependent variables (u1,v1,u2,v2,p1,p2) from the vector
        u1 = x[0:N]
        v1 = x[N:2*N]
        u2 = x[2*N:3*N]
        v2 = x[3*N:4*N]
        p1 = x[4*N:5*N]
        p2 = x[5*N:6*N]
        #Create some additional vectors
        Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        ks = np.ravel(Ks)
        ls = np.ravel(Ls)
        sigma1 = self.abl.U1*ks+self.abl.V1*ls
        sigma2 = self.abl.U2*ks+self.abl.V2*ls
        #Regular entries
        #u1 equation
        Axu1 += (-1j*sigma1
                -2*self.abl.C1*self.abl.dS12/self.abl.H1
                -2*self.abl.C0*self.abl.S1/self.abl.H1)*u1
        Axu1 += self.abl.fc*v1
        Axu1 += 2*self.abl.C1*self.abl.dS12/self.abl.H1*u2
        Axu1 += -1j*ks*(p1+p2)
        #v1 equation
        Axv1 += -self.abl.fc*u1
        Axv1 += (-1j*sigma1
                -2*self.abl.C1*self.abl.dS12/self.abl.H1
                -2*self.abl.C0*self.abl.S1/self.abl.H1)*v1
        Axv1 += 2*self.abl.C1*self.abl.dS12/self.abl.H1*v2
        Axv1 += -1j*ls*(p1+p2)
        #u2 equation
        Axu2 += 2*self.abl.C1*self.abl.dS12/self.abl.H2*u1
        Axu2 += (-1j*sigma2
                -2*self.abl.C1*self.abl.dS12/self.abl.H2)*u2
        Axu2 += self.abl.fc*v2
        Axu2 += -1j*ks*(p1+p2)
        #v2 equation
        Axv2 += 2*self.abl.C1*self.abl.dS12/self.abl.H2*v1
        Axv2 += -self.abl.fc*u2
        Axv2 += (-1j*sigma2
                -2*self.abl.C1*self.abl.dS12/self.abl.H2)*v2
        Axv2 += -1j*ls*(p1+p2)
        #p1 equation
        #General case
        Axp1 += np.ravel(self.PHI)*self.abl.H1*ks*u1
        Axp1 += np.ravel(self.PHI)*self.abl.H1*ls*v1
        Axp1 += sigma1*p1
        #p2 equation
        #General case
        Axp2 += np.ravel(self.PHI)*self.abl.H2*ks*u2
        Axp2 += np.ravel(self.PHI)*self.abl.H2*ls*v2
        Axp2 += sigma2*p2
        #p1,2 equation: k=l=0
        indices_klzero = int(Nx/2)*Ny+int(Ny/2)
        Axp1[indices_klzero] = p1[indices_klzero]
        Axp2[indices_klzero] = p2[indices_klzero]

        #Compute 1st order forcing term
        u1r = self.c2r(u1.reshape(self.grid.shape))
        v1r = self.c2r(v1.reshape(self.grid.shape))
        F1u, F1v = self.forcing.F1(self.abl,u1r,v1r)
        #Convert back to fourier space, cast into 1D array and
        #divide by H1 (TLM solves height-averaged equations)
        Axu1 += -np.ravel(self.r2c(F1u))/self.abl.H1
        Axv1 += -np.ravel(self.r2c(F1v))/self.abl.H1

        #Set defunct modes to zero
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        Axu1[defunctindices] = 0.
        Axv1[defunctindices] = 0.
        Axu2[defunctindices] = 0.
        Axv2[defunctindices] = 0.
        Axp1[defunctindices] = 0.
        Axp2[defunctindices] = 0.
        return np.concatenate((Axu1,Axv1,Axu2,Axv2,Axp1,Axp2))

class U1Dmodel(model):
#No longer up to date
    '''
    Unsteady one-dimensional gravity wave model
    '''
#    #Real in space but spectral in time
#    result['u1f']  = np.fft.ifft(np.fft.ifftshift(result['u1c']*Nx,axes=0),axis=0)

    def c2r(self,cfield,returnErr=False):
        rfield = np.fft.ifft2(np.fft.ifftshift(cfield*self.grid.N))
        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def buildABmatrix(self):
        Nx = self.grid.Nx
        Nt = self.grid.Nt
        A = np.zeros((4*Nx*Nt,4*Nx*Nt),dtype=np.complex128)
        B = np.zeros((4*Nx*Nt),dtype=np.complex128)
        PHIs = np.zeros((Nx,Nt),dtype=np.complex128)
        for indexk, k in enumerate(self.grid.ks):
            for indexo, omega in enumerate(self.grid.omegas):
                index = indexo + Nt*indexk
                sigma1 = self.abl.U1*k-omega
                sigma2 = self.abl.U2*k-omega
                sigma3 = self.abl.U3*k-omega
        
                #Hydrostatic solution 
                PHI = self.abl.gprime
                if not sigma3==0 and not k==0:
                    m = np.sign(k)*k*self.abl.N/sigma3
                    PHI += 1j/m*(self.abl.N**2)
                PHIs[indexk,indexo] = PHI
        
                #B matrix
                B[index]= self.wf.CTr[indexk,indexo]*self.abl.S1*self.abl.U1/self.abl.H1
                B[index+Nx*Nt]= self.wf.CTc[indexk,indexo]*self.abl.S1*self.abl.V1/self.abl.H1
                
                #A matrix regular entries
                #u1 equation
                A[index,index] += -1j*sigma1
                A[index,index] +=  1j*k*PHI*self.abl.H1/self.abl.U1
                A[index,index] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
                A[index,index] += -2*self.abl.C0*self.abl.S1/self.abl.H1
                A[index,index+1*Nx*Nt] += self.abl.fc
                A[index,index+2*Nx*Nt] += 1j*k*PHI*self.abl.H2/self.abl.U2
                A[index,index+2*Nx*Nt] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
        
                #v1 equation
                A[index+Nx*Nt,index] += -self.abl.fc
                A[index+Nx*Nt,index+1*Nx*Nt] += -1j*sigma1
                A[index+Nx*Nt,index+1*Nx*Nt] += -2*self.abl.C1*self.abl.dS12/self.abl.H1
                A[index+Nx*Nt,index+1*Nx*Nt] += -2*self.abl.C0*self.abl.S1/self.abl.H1
                A[index+Nx*Nt,index+3*Nx*Nt] += 2*self.abl.C1*self.abl.dS12/self.abl.H1
        
                #u2 equation
                A[index+2*Nx*Nt,index] += 1j*k*PHI*self.abl.H1/self.abl.U1
                A[index+2*Nx*Nt,index] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
                A[index+2*Nx*Nt,index+2*Nx*Nt] += -1j*sigma2
                A[index+2*Nx*Nt,index+2*Nx*Nt] +=  1j*k*PHI*self.abl.H2/self.abl.U2
                A[index+2*Nx*Nt,index+2*Nx*Nt] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
                A[index+2*Nx*Nt,index+3*Nx*Nt] += self.abl.fc
        
                #v2 equation
                A[index+3*Nx*Nt,index+1*Nx*Nt] += 2*self.abl.C1*self.abl.dS12/self.abl.H2
                A[index+3*Nx*Nt,index+2*Nx*Nt] += -self.abl.fc
                A[index+3*Nx*Nt,index+3*Nx*Nt] += -1j*sigma2
                A[index+3*Nx*Nt,index+3*Nx*Nt] += -2*self.abl.C1*self.abl.dS12/self.abl.H2
        A[:Nx*Nt,:Nx*Nt] += -2*self.abl.S1/self.abl.H1*cconv_2D(self.wf.CTc)
        A[Nx*Nt:2*Nx*Nt,Nx*Nt:2*Nx*Nt] += -2*self.abl.S1/self.abl.H1*cconv_2D(self.wf.CTc)
    
        #Set defunct modes to zero
        defunct_k = [i for i in range(Nt)]
        defunct_o = [i*Nt for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_o)
        A[defunctindices,:]    = 0.
        A[defunctindices+Nx*Nt,:]   = 0.
        A[defunctindices+2*Nx*Nt,:] = 0.
        A[defunctindices+3*Nx*Nt,:] = 0.
        A[defunctindices,defunctindices]                 = 1.
        A[defunctindices+Nx*Nt,defunctindices+Nx*Nt]     = 1.
        A[defunctindices+2*Nx*Nt,defunctindices+2*Nx*Nt] = 1.
        A[defunctindices+3*Nx*Nt,defunctindices+3*Nx*Nt] = 1.
        B[defunctindices]    = 0.
        B[defunctindices+Nx*Nt]   = 0.
        B[defunctindices+2*Nx*Nt] = 0.
        B[defunctindices+3*Nx*Nt] = 0.
        
        return A,B,PHIs

class ABL(object):
    '''
    Atmospheric state

    Data structure containing all information about the atmospheric state
    '''
    def __init__(self,input='LESbased',**kwargs):
        '''
        Parameters
        ----------
        input: str
            Name of the method used to specify the atmospheric state
        '''
        #input flag indicates how the data is specified
        assert input in ['default_subcr',
                         'default_supercr',
                         'LESbased',
                         'analytic_constant',
                         'analytic_quadratic',
                         'analytic_cubic',
                         'ERA5',
                         'fromfile'],'Error: ABL input mode unknown'
        self.__nu1 = 0.
        self.__nu2 = 0.
        self.__zs = None
        self.__us = None
        self.__vs = None
        self.__zst = None
        self.__ths = None
        self.__taus = None
        self.__TIs = None
        self.__TI = 0.12
        
        function = getattr(self,input)
        function(**kwargs)

    def default_supercr(self,**kwargs):
        #Default supercritical abl state, corresponding to finWF5
        self.__H1 = 150.0
        self.__U1 = 10.27
        self.__V1 = 0.0588
        self.__H2 = 905.0
        self.__U2 = 11.97
        self.__V2 = -0.811
        self.__U3 = 11.888
        self.__V3 = -1.633
        self.__C0 = 0.91437755e-3
        self.__C1 = 1.89040249e-2
        self.__C2 = 1.73034702e-3
        self.__gprime = 3.432e-2
        self.__fc     = 1.0e-4
        self.__N      = 0.58354e-2

    def default_subcr(self,**kwargs):
        #Default subcritical abl state, corresponding to finSBL_q00
        self.__H1 = 150.0
        self.__U1 = 8.5518
        self.__V1 = 0.00299
        self.__H2 = 910.0
        self.__U2 = 11.995
        self.__V2 = -1.5331
        self.__U3 = 11.505
        self.__V3 = -3.4106
        self.__C0 = 0.0034569626
        self.__C1 = 0.0145373837
        self.__C2 = 0.0005207619
        self.__gprime = 0.1696
        self.__fc     = 1.0e-4
        self.__N      = 0.58132e-2

    def LESbased(self,**kwargs):
        arguments = ['sim','tstart','tend','H1',
                     'ccfilename','stfilename',
                     'EKfilename','ENfilename']
        assert all([i in kwargs for i in arguments]),'Error: some arguments for LESbased ABL definition are missing'

        sim = kwargs['sim']
        #Load utau and vertical profiles from BL_tstatcc and BL_tstatst
        data = loadsp.BLtstatall(kwargs['ccfilename'],
                                 kwargs['stfilename'],
                                 sim.Lref,sim.Uref)
        f = interpolate.interp1d(data['z'],data['uw'],fill_value='extrapolate')
        uwst = f(data['zst'])
        f = interpolate.interp1d(data['z'],data['vw'],fill_value='extrapolate')
        vwst = f(data['zst'])
        tau13 = -uwst - data['s13']
        tau23 = -vwst - data['s23']
        tau = np.sqrt(tau13**2 + tau23**2)
        self.__zs = data['z']
        self.__zst = data['zst']
        self.__us = data['u']
        self.__vs = data['v']

        #Turbulent intensity at hub height
        f = interpolate.interp1d(data['z'],
                    np.sqrt((data['uu']+data['vv']+data['ww'])/3.0)/self.Ms)
        self.__TI = np.asscalar(f(kwargs['H1']/2.0))
    
        #Load geostrophic wind angle from ek_post
        EKp = loadsp.EKpost(kwargs['EKfilename'],sim.Lref,sim.Uref)
        istart = np.max(np.where(EKp['t']<=kwargs['tstart']))
        iend = np.max(np.where(EKp['t']<=kwargs['tend']))
        alpha = np.mean(EKp['alpha'][istart:iend+1])
        self.__U3 = sim.abl.G*np.cos(alpha)
        self.__V3 = sim.abl.G*np.sin(alpha)
        
        #Estimate inversion parameters
        #filename = os.path.join(sim.path,'precursor','en_tstatcc_precursor.dat')
        ENdata = loadsp.ENtstatcc(kwargs['ENfilename'],
                                    sim.Lref,sim.Uref,sim.Tref)
        CIestimate = CIops.RZfit(ENdata['z'],ENdata['th'],
                                    [0.9,0.1,sim.Tref,1000.0,100.0])
    
        #Compute height averaged quantities
        #Layer 1
        i1 = np.max(np.where(data['zst']<=kwargs['H1']))-1
        self.__H1 = data['zst'][i1+1]-data['zst'][0]
        self.__U1 = mypy.trapzst(data['u'],data['zst'],0,i1)/self.H1
        self.__V1 = mypy.trapzst(data['v'],data['zst'],0,i1)/self.H1
        tau01 = data['utau']**2
        self.__C0 = tau01/self.S1**2
        #Layer 2
        i2 = np.max(np.where(data['zst']<=CIestimate['h1']))-1
        self.__H2 = data['zst'][i2+1]-data['zst'][i1+1]
        self.__U2 = mypy.trapzst(data['u'],data['zst'],i1+1,i2)/self.H2
        self.__V2 = mypy.trapzst(data['v'],data['zst'],i1+1,i2)/self.H2
        tau12 = tau[i1+1]
        tau23 = tau[i2+1]
        self.__C1 = tau12/self.dS12**2
        self.__C2 = tau23/self.dS23**2
    
        #Other ABL parameters
        self.__gprime = sim.abl.gravity*CIestimate['dth']/sim.Tref
        self.__N  = np.sqrt(sim.abl.gravity*CIestimate['gamma']/sim.Tref)
        self.__fc = sim.abl.fc

    def analytic_constant(self,**kwargs):
        #ABL state based on analytical formulas for u and v (Csanady 1974)
        arguments = ['dth','fc','N','G','alpha','viscosity','utau','h']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_constant ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__fc = kwargs['fc']
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = 150.0
        self.__H2 = kwargs['h']-self.H1

        h = kwargs['h']
        K = kwargs['viscosity']
        utau = kwargs['utau']

        #Velocity deficit vector
        Nz = 100
        zst = np.linspace(0.,h,Nz)
        zs = (zst[0:-1]+zst[1:])/2.0
        gamma = (1j+1) * (2*K/(self.fc*h**2))**(-1/2)
        wd = (1j-1)* utau**2/(2*K*self.fc) * np.cosh(gamma*(zs/h-1))/np.sinh(gamma)
        dwdz = (1j-1)* utau**2/(2*K*self.fc) * np.sinh(gamma*(zs/h-1))/np.sinh(gamma) * gamma/h

        u = self.U3+np.real(wd)*utau
        v = self.V3+np.imag(wd)*utau
        dudz = np.real(dwdz)*utau
        dvdz = np.imag(dwdz)*utau
        tau = K*np.sqrt(dudz**2+dvdz**2)
        #Compute height averaged quantities
        #Layer 1
        i1 = np.max(np.where(zst<=self.H1))-1
        self.__U1 = mypy.trapzst(u,zst,0,i1)/self.H1
        self.__V1 = mypy.trapzst(v,zst,0,i1)/self.H1
        self.__nu1 = K
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2
        #Layer 2
        self.__U2 = mypy.trapzst(u,zst,i1+1,Nz)/self.H2
        self.__V2 = mypy.trapzst(v,zst,i1+1,Nz)/self.H2
        self.__nu2 = K
        tau12 = tau[i1+1]
        self.__C1 = tau12/self.dS12**2
        self.__C2 = 0.

    def analytic_quadratic(self,**kwargs):
        #ABL state based on analytical formulas for u and v (Nieuwstadt 1983)
        arguments = ['dth','fc','N','G','alpha','kappa','utau','h']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_quadratic ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__fc = kwargs['fc']
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = 150.0
        self.__H2 = kwargs['h']-self.H1

        h = kwargs['h']
        kappa = kwargs['kappa']
        utau = kwargs['utau']

        #Velocity deficit vector
        Nz = 100
        zst = np.linspace(0.,h,Nz)
        zs = (zst[0:-1]+zst[1:])/2.0
        C = h*self.fc/kappa/utau
        beta = 0.5*np.sqrt(1-4j*C)
        wd = np.zeros((Nz-1),dtype=np.complex128)
        sigma = np.zeros((Nz-1),dtype=np.complex128)
        for k in range(Nz-1):
            wd[k] = -np.pi/(kappa*np.cos(np.pi*beta))*mpmath.hyp2f1(0.5+beta,
                0.5-beta,1,1-zs[k]/h)
            sigma[k] = 1j*np.pi*C/np.cos(np.pi*beta)*(1-zs[k]/h)*mpmath.hyp2f1(0.5+beta,0.5-beta,2,1-zs[k]/h)

        u = self.U3+np.real(wd)*utau
        v = self.V3+np.imag(wd)*utau
        taux = np.real(sigma)*utau**2
        tauy = np.imag(sigma)*utau**2
        tau = np.sqrt(taux**2+tauy**2)
        #Compute height averaged quantities
        #Layer 1
        i1 = np.max(np.where(zst<=self.H1))-1
        self.__U1 = mypy.trapzst(u,zst,0,i1)/self.H1
        self.__V1 = mypy.trapzst(v,zst,0,i1)/self.H1
        self.__nu1 = mypy.trapzst(kappa*utau*zs*(1-zs/h),zst,0,i1)/self.H1
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2
        #Layer 2
        self.__U2 = mypy.trapzst(u,zst,i1+1,Nz)/self.H2
        self.__V2 = mypy.trapzst(v,zst,i1+1,Nz)/self.H2
        self.__nu2 = mypy.trapzst(kappa*utau*zs*(1-zs/h),zst,i1+1,Nz)/self.H2
        tau12 = tau[i1+1]
        self.__C1 = tau12/self.dS12**2
        self.__C2 = 0.

    def analytic_cubic(self,**kwargs):
        #ABL state based on analytical formulas for u and v (Nieuwstadt 1983)
        arguments = ['dth','fc','N','G','alpha','kappa','utau','h','H1']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_quadratic ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__fc = kwargs['fc']
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = kwargs['H1']
        self.__H2 = kwargs['h']-self.H1

        h = kwargs['h']
        kappa = kwargs['kappa']
        utau = kwargs['utau']

        #Velocity deficit vector
        Nz = 100
        self.__zst = np.linspace(0.,h,Nz)
        self.__zs = (self.zst[0:-1]+self.zst[1:])/2.0
        C = h*self.fc/kappa/utau
        alpha = 0.5+0.5*np.sqrt(1+4j*C)
        wd = np.zeros((Nz-1),dtype=np.complex128)
        sigma = np.zeros((Nz-1),dtype=np.complex128)
        for k in range(Nz-1):
            wd[k] = (1j*alpha**2*(mpmath.gamma(alpha))**2)/(kappa*C*mpmath.gamma(2*alpha))*(1-self.zs[k]/h)**(alpha-1)*mpmath.hyp2f1(alpha+1,alpha-1,2*alpha,1-self.zs[k]/h)
            sigma[k] = (alpha*(mpmath.gamma(alpha))**2)/(mpmath.gamma(2*alpha))*(1-self.zs[k]/h)**(alpha)*mpmath.hyp2f1(alpha-1,alpha,2*alpha,1-self.zs[k]/h)

        self.__us = self.U3+np.real(wd)*utau
        self.__vs = self.V3+np.imag(wd)*utau
        taux = np.real(sigma)*utau**2
        tauy = np.imag(sigma)*utau**2
        tau = np.sqrt(taux**2+tauy**2)
        #Compute height averaged quantities
        #Layer 1
        i1 = np.max(np.where(self.zst<=self.H1))-1
        self.__U1 = mypy.trapzst(self.us,self.zst,0,i1)/self.H1
        self.__V1 = mypy.trapzst(self.vs,self.zst,0,i1)/self.H1
        self.__nu1 = mypy.trapzst(kappa*utau*self.zs*(1-self.zs/h)**2,self.zst,0,i1)/self.H1
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2
        #Layer 2
        self.__U2 = mypy.trapzst(self.us,self.zst,i1+1,Nz)/self.H2
        self.__V2 = mypy.trapzst(self.vs,self.zst,i1+1,Nz)/self.H2
        self.__nu2 = mypy.trapzst(kappa*utau*self.zs*(1-self.zs/h)**2,self.zst,i1+1,Nz)/self.H2
        tau12 = tau[i1+1]
        self.__C1 = tau12/self.dS12**2
        self.__C2 = 0.

    def ERA5(self,**kwargs):
        arguments = ['H1','T2','blh','ust','wth','phi',
                     'zs','us','vs','ths','Gmode']
        assert all([i in kwargs for i in arguments]),'Error: some arguments for ERA5 based ABL definition are missing'

        gravity = 9.80665    # [m s-2]
        P0 = 1.e5 # Reference pressure [Pa]
        R_air = 287.058 # Specific gas constant for dry air [J kg-1 K-1]
        Cp_air = 1005   # Specific heat of air [J kg-1 K-1]
        kappa  = 0.41   # Von Karman constant
        eps = 0.609133  # Rv/Rd-1
        omega = 7.2921159e-5    # angular speed of the Earth [rad/s]

        #Surface parameters
        ust = kwargs['ust']
        wth = kwargs['wth']
        T2  = kwargs['T2']
        blh = kwargs['blh']
        zeta = -2.0*kappa*gravity*wth/(T2*ust**3)

        #Vertical profiles (arrays are in reversed order)
        self.__zs  = kwargs['zs'][::-1]
        self.__us  = kwargs['us'][::-1]
        self.__vs  = kwargs['vs'][::-1]
        self.__ths = kwargs['ths'][::-1]

        #Estimate inversion parameters
        zCI = self.zs[self.zs<5000]
        thCI = self.ths[self.zs<5000]
        if 'dh_max' in kwargs:
            dh_max = kwargs['dh_max']
        else:
            dh_max=None

        if zeta>0.02:
            #Ignore temperature decrease inside SBL
            #(we are trying to indentify the mixing layer that preceded this SBL)
            thCI[zCI<blh] = interpolate.interp1d(zCI,thCI)(blh)
            CIestimate = CIops.RZfit(zCI,thCI,p0=[0.9,0.1,T2,1000.,100.0],
                                            dh_max=dh_max)
        else:
            #Ignore temperature increase in CBL surface layer
            thCI[0:np.argmin(thCI)] = np.min(thCI)
            CIestimate = CIops.RZfit(zCI,thCI,p0=[0.9,0.1,T2,blh,100.0],
                                            dh_max=dh_max)

        #No inversion strength in the following cases:
        #a<=0.2: encroachment
        #a<=2*b: inversion lapse rate is equal to or smaller than free lapse rate
        if (CIestimate['a']<=0.2 or CIestimate['a']<=2*CIestimate['b']):
            H = np.max([CIestimate['h1'],kwargs['H1']+10])
            self.__gprime = 0.
        else:
            H = np.max([CIestimate['h1'],kwargs['H1']+10])
            self.__gprime = gravity*CIestimate['dth']/T2

        #Flux profile
        tau    = np.zeros(self.zs.shape)
        nu     = np.zeros(self.zs.shape)
        if zeta>0.0:
            tau[self.zs<=blh] = ust**2*(1-self.zs[self.zs<=blh]/blh)**(1.5)
            nu[self.zs<=blh]  = kappa*ust*self.zs[self.zs<=blh]*(1-self.zs[self.zs<=blh]/blh)**2
        else:
            tau[self.zs<=H] = ust**2*(1-self.zs[self.zs<=H]/H)
            nu[self.zs<=H]  = kappa*ust*self.zs[self.zs<=H]*(1-self.zs[self.zs<=H]/H)**2
        self.__taus = tau

        fu = interpolate.interp1d(self.zs,self.us,fill_value='extrapolate')
        fv = interpolate.interp1d(self.zs,self.vs,fill_value='extrapolate')
        ft = interpolate.interp1d(self.zs,self.taus,fill_value='extrapolate')
        fn = interpolate.interp1d(self.zs,nu,fill_value='extrapolate')
        #Compute height averaged quantities
        #Layer 1
        self.__H1 = kwargs['H1']
        z = np.linspace(0,self.H1,100)
        self.__U1 = np.trapz(fu(z),z)/self.H1
        self.__V1 = np.trapz(fv(z),z)/self.H1
#        self.__nu1 = np.trapz(fn(z),z)/self.H1
        #Layer 2
        self.__H2 = H-self.H1
        z = np.linspace(self.H1,self.H,100)
        self.__U2 = np.trapz(fu(z),z)/self.H2
        self.__V2 = np.trapz(fv(z),z)/self.H2
#        self.__nu2 = np.trapz(fn(z),z)/self.H2
        #Layer 3
        if kwargs['Gmode'] == 'h1':
            self.__U3 = np.asscalar(fu(CIestimate['h1']))
            self.__V3 = np.asscalar(fv(CIestimate['h1']))
        elif kwargs['Gmode'] == 'h2':
            self.__U3 = np.asscalar(fu(CIestimate['h2']))
            self.__V3 = np.asscalar(fv(CIestimate['h2']))
        elif kwargs['Gmode'] == 'top':
            self.__U3 = np.asscalar(fu(5000.))
            self.__V3 = np.asscalar(fv(5000.))
        elif kwargs['Gmode'] == 'avg':
            z = np.linspace(self.H,5000.,1000)
            self.__U3 = np.trapz(fu(z),z)/(z[-1]-z[0])
            self.__V3 = np.trapz(fu(z),z)/(z[-1]-z[0])
        else:
            print('Gmode unknown, abort')
            return 1

        #Stress levels
        tau01 = ust**2
        tau12 = ft(self.H1)
        tau23 = ft(self.H)
        self.__C0 = tau01/self.S1**2
        self.__C1 = tau12/self.dS12**2
        self.__C2 = tau23/self.dS23**2

        #Turbulent intensity at hub height
        if zeta>0.0:
            #From Nieuwstadt (1984): q/sqrt(tau) = 3
            tke = 4.5*tau
            f = interpolate.interp1d(self.zs,np.sqrt(2./3.*tke/self.Ms))
        else:
            #From Stull (1988): q^2/tau = 8.5+2.5
            tke = 5.5*tau
            f = interpolate.interp1d(self.zs,np.sqrt(2./3.*tke/self.Ms))
        self.__TI = np.asscalar(f(self.H1/2.0))
        self.__TIs = f(self.zs)
    
        #Other ABL parameters
        self.__N  = np.sqrt(gravity*CIestimate['gamma']/T2)
        self.__fc = 2*omega*np.sin(kwargs['phi'])

    def fromfile(self,**kwargs):
        #load ABL state from file
        assert 'filename' in kwargs, 'Error: filename not specified'
        
        #Read from file
        with open(kwargs['filename'],'r') as file:
            for _ in range(7): file.readline()
            self.__H1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__U1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__V1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__nu1    = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__H2     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__U2     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__V2     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__nu2    = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__U3     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__V3     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__C0     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__C1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__C2     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__gprime = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__N      = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__fc     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__TI     = float(file.readline().rstrip('\r\n').split('=')[1])
            file.readline()
            #If not yet at end of file, continue reading zs, us, vs and zst
            if file.readline() is not '':
                file.readline()
                Nz = int(file.readline().rstrip('\r\n').split('=')[1])
                file.readline()
                dummy = []
                for _ in range(Nz):
                    dummy.append([float(i) for i in file.readline().rstrip('\r\n').split(',')])
                data = np.array(dummy)
                self.__zs = data[:,0]
                self.__us = data[:,1]
                self.__vs = data[:,2]
                for _ in range(4): file.readline()
                dummy = []
                for _ in range(Nz+1):
                    dummy.append(float(file.readline().rstrip('\r\n')))
                self.__zst = np.array(dummy)
            else:
                self.__zs = None
                self.__us = None
                self.__vs = None
                self.__zst = None

    def rotate(self,alpha):
        U1n = self.U1*np.cos(alpha)+self.V1*np.sin(alpha)
        V1n = self.V1*np.cos(alpha)-self.U1*np.sin(alpha)
        self.__U1 = U1n
        self.__V1 = V1n
        U2n = self.U2*np.cos(alpha)+self.V2*np.sin(alpha)
        V2n = self.V2*np.cos(alpha)-self.U2*np.sin(alpha)
        self.__U2 = U2n
        self.__V2 = V2n
        U3n = self.U3*np.cos(alpha)+self.V3*np.sin(alpha)
        V3n = self.V3*np.cos(alpha)-self.U3*np.sin(alpha)
        self.__U3 = U3n
        self.__V3 = V3n

    def kwake(self,TI=None):
        if not TI:
            TI = self.TI
        return 0.3837*TI+0.003678

    def saveas(self,filename,info=''):
        with open(filename,'w') as file:
            file.write('%%%%%%%%%%%%%%\n')
            file.write('TLM ABL object\n')
            file.write('%%%%%%%%%%%%%%\n')
            file.write(info+'\n')
            file.write('\n')
            file.write('1. Scalar data\n')
            file.write('--------------\n')
            file.write('H1     ='+'{:17.10g}'.format(self.H1)+'\n')
            file.write('U1     ='+'{:17.10g}'.format(self.U1)+'\n')
            file.write('V1     ='+'{:17.10g}'.format(self.V1)+'\n')
            file.write('nu1    ='+'{:17.10g}'.format(self.nu1)+'\n')
            file.write('H2     ='+'{:17.10g}'.format(self.H2)+'\n')
            file.write('U2     ='+'{:17.10g}'.format(self.U2)+'\n')
            file.write('V2     ='+'{:17.10g}'.format(self.V2)+'\n')
            file.write('nu2    ='+'{:17.10g}'.format(self.nu2)+'\n')
            file.write('U3     ='+'{:17.10g}'.format(self.U3)+'\n')
            file.write('V3     ='+'{:17.10g}'.format(self.V3)+'\n')
            file.write('C0     ='+'{:17.10g}'.format(self.C0)+'\n')
            file.write('C1     ='+'{:17.10g}'.format(self.C1)+'\n')
            file.write('C2     ='+'{:17.10g}'.format(self.C2)+'\n')
            file.write('gprime ='+'{:17.10g}'.format(self.gprime)+'\n')
            file.write('N      ='+'{:17.10g}'.format(self.N)+'\n')
            file.write('fc     ='+'{:17.10g}'.format(self.fc)+'\n')
            file.write('TI     ='+'{:17.10g}'.format(self.TI)+'\n')

            if self.us is not None:
                file.write('\n')
                file.write('2. Cell-centered data\n')
                file.write('---------------------\n')
                file.write('Nz ='+'{:3d}'.format(self.zs.size)+'\n')
                file.write('zs,us,vs\n')
                for i in range(self.us.size):
                    file.write('{:17.10g}'.format(self.zs[i])+','
                               '{:17.10g}'.format(self.us[i])+','
                               '{:17.10g}'.format(self.vs[i])+'\n')
                file.write('\n')
                file.write('3. Staggered data\n')
                file.write('---------------------\n')
                file.write('zst\n')
                for i in range(self.zst.size):
                    file.write('{:17.10g}'.format(self.zst[i])+'\n')


    @property
    def us(self):
        '''Velocity profile in dimension 0 used to derive
        height-averaged velocities'''
        return self.__us
    @property
    def vs(self):
        '''Velocity profile in dimension 1 used to derive
        height-averaged velocities'''
        return self.__vs
    @property
    def Ms(self):
        '''Velocity magnitude profile (for post-processing purposes)'''
        return np.sqrt(self.us**2+self.vs**2)
    @property
    def ths(self):
        '''Potential temperature profile used to estimate
        temperature structure'''
        return self.__ths
    @property
    def taus(self):
        '''Shear stress profile used to estimate
        momentum transport coefficients'''
        return self.__taus
    @property
    def TIs(self):
        '''Turbulent intensity profile used to estimate
        TI at hub height'''
        return self.__TIs
    @property
    def zs(self):
        '''Height corresponding to the vertical profiles (at cell centers)'''
        return self.__zs
    @property
    def zst(self):
        '''Height corresponding to the vertical profiles (at cell faces)'''
        return self.__zst
    @property
    def H1(self):
        '''Height of the wind-farm layer'''
        return self.__H1
    @property
    def U1(self):
        '''Height-averaged velocity in the wind-farm layer in dimension 0'''
        return self.__U1
    @U1.setter
    def U1(self,value):
        self.__U1 = value
    @property
    def V1(self):
        '''Height averaged velocity in the wind-farm layer in dimension 1'''
        return self.__V1
    @V1.setter
    def V1(self,value):
        self.__V1 = value
    @property
    def S1(self):
        '''Height-averaged velocity magnitude in the wind-farm layer'''
        return np.sqrt(self.U1**2 + self.V1**2)
    @property
    def WD1(self):
        '''
        Height-averaged wind direction in the wind-farm layer (degrees)

        Bug: np.arctan only recognises angles between -90 and +90
        Better would be to return np.arctan2(self.V1,self.U1)*180/np.pi
        Even better is to return the actual wind direction:
            return 180. + np.arctan2(self.U1,self.V1)*180/np.pi
        '''
        return np.arctan(self.V1/self.U1)*180/np.pi
    @property
    def nu1(self):
        '''Height-averaged turbulent viscosity in the wind-farm layer'''
        return self.__nu1
    @nu1.setter
    def nu1(self,value):
        self.__nu1 = value
    @property
    def H2(self):
        '''Height of the upper layer'''
        return self.__H2
    @property
    def U2(self):
        return self.__U2
    @property
    def V2(self):
        return self.__V2
    @V2.setter
    def V2(self,value):
        self.__V2 = value
    @property
    def S2(self):
        return np.sqrt(self.U2**2 + self.V2**2)
    @property
    def WD2(self):
        '''
        Height-averaged wind direction in the upper layer (degrees)

        Bug: see WD1
        '''
        return np.arctan(self.V2/self.U2)*180/np.pi
    @property
    def nu2(self):
        return self.__nu2
    @nu2.setter
    def nu2(self,value):
        self.__nu2 = value
    @property
    def U3(self):
        return self.__U3
    @property
    def V3(self):
        return self.__V3
    @V3.setter
    def V3(self,value):
        self.__V3 = value
    @property
    def S3(self):
        return np.sqrt(self.U3**2 + self.V3**2)
    @property
    def WD3(self):
        '''
        Wind direction in the free atmosphere (degrees)

        Bug: see WD1
        '''
        return np.arctan(self.V3/self.U3)*180/np.pi
    @property
    def dU12(self):
        return self.U1-self.U2
    @property
    def dV12(self):
        return self.V1-self.V2
    @property
    def dS12(self):
        return np.sqrt(self.dU12**2 + self.dV12**2)
    @property
    def dS23(self):
        return np.sqrt((self.U3-self.U2)**2 + (self.V3-self.V2)**2)
    @property
    def C0(self):
        return self.__C0
    @property
    def C1(self):
        return self.__C1
    @property
    def C2(self):
        return self.__C2
    @property
    def gprime(self):
        return self.__gprime
    @gprime.setter
    def gprime(self,value):
        self.__gprime = value
    @property
    def N(self):
        return self.__N
    @property
    def fc(self):
        return self.__fc
    @fc.setter
    def fc(self,value):
        self.__fc = value
    @property
    def TI(self):
        '''Streamwise turbulent intensity at hub height'''
        return self.__TI
    @TI.setter
    def TI(self,value):
        self.__TI = value
    @property
    def H(self):
        return self.H1+self.H2
    @property
    def Ub(self):
        #Projection of (U2,V2) onto (U1,V1)
        U2p = (self.U1*self.U2+self.V1*self.V2)/self.S1
        return (self.H1/self.H*self.S1**(-2)+self.H2/self.H*U2p**(-2))**(-1/2)
    @property
    def PN(self):
        return self.Ub**2/(self.N*self.S3*self.H)
    @property
    def Fr(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return self.Ub/np.sqrt(self.gprime*self.H)
    @property
    def Fr1(self):
        return self.S1/np.sqrt(self.gprime*self.H1)
    @property
    def Fr2(self):
        return self.S2/np.sqrt(self.gprime*self.H2)

def cconv_1D(b,method='Fast'):
    '''
    Compose a matrix that expresses the 1D circular convolution
    as a matrix vector multiplication (conv(a,b) = Ba)
    
    b is assumed to be a vector with wave numbers centered around 0
    (the 0 mode corresponds to b[int(b.shape[0]/2)] )
    '''
    if method=='Naive':
        #Naive method
        N = b.shape[0]
        B = np.zeros((N,N))
        for k in range(N):
            for k2 in range(N):
                kshifted = k-k2+int(N/2)
                if kshifted>=N:
                    kshifted -= N
                B[k,k2] += b[kshifted]
    elif method=='Fast':
        #Fast method using toeplitz matrix function
        c = np.fft.ifftshift(b)
        r = np.append(np.array(c[0]),np.flipud(c[1:]))
        B = scipy.linalg.toeplitz(c,r)

    return B

def cconv_2D(b,method='Fast'):
    '''
    Compose a matrix that expresses the 2D circular convolution
    as a matrix vector multiplication (conv(a,b) = Ba)
    
    b is assumed to be a matrix with wave numbers centered around 0
    (the 0 mode corresponds to b[int(b.shape[0]/2),int(b.shape[1]/2] )
    '''
    Nk = b.shape[0]
    Nl = b.shape[1]
    if method=='Naive':
        #Naive method
        B = np.zeros((Nk*Nl,Nk*Nl),dtype=np.complex128)
        for k in range(Nk):
            for l in range(Nl):
                index = l + Nl*k
                for k2 in range(Nk):
                    for l2 in range(Nl):
                        kshifted = k-k2+int(Nk/2)
                        lshifted = l-l2+int(Nl/2)
                        if kshifted>=Nk:
                            kshifted -= Nk
                        if lshifted>=Nl:
                            lshifted -= Nl
                        index2 = l2 + Nl*k2
                        B[index,index2] += b[kshifted,lshifted]

    elif method=='Fast':
        #Fast method
        bshift = np.fft.ifftshift(b)
        #Construct an array with the Nk different block toeplitz matrices
        #(undo the fftshift on the 1D input vector as this is also done in cconv_1D)
        c = np.array([cconv_1D(np.fft.fftshift(bshift[i,:])) for i in range(Nk)])
        #Construct amtrix with toeplitz structure,
        #but inputs are now block matrices instead of scalars
        #cpied from implementation of scipy.linalg.toeplitz)
        vals = np.concatenate((c[1:,:,:],c))
        i1, i2 = np.ogrid[0:len(c), len(c) - 1:-1:-1]
        A = vals[i1+i2]
        #shape of A is (Nk,Nk,Nl,Nl), revert to (Nk*Nl,Nk*Nl) matrix
        x = np.swapaxes(A,1,2).reshape(Nk,Nl,Nk*Nl)
        B = x.reshape(Nk*Nl,Nk*Nl)
    return B

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class lgmres_counter(object):
    def __init__(self,A,B,N,disp=True):
        self._disp = disp
        self.niter = 0
        self.counter = 0
        self.A = A
        self.B = B
        self.N = N
        self.residu = np.zeros((N))
        if self._disp:
            print('1----------------------------------------{}'.format(self.N))
    def __call__(self, xk=None):
        self.niter += 1
        self.residu[self.niter] = np.linalg.norm(self.A.matvec(xk)-self.B)
        dn = round(self.niter/self.N*40)
        ni = dn-self.counter
        if self._disp:
            for i in range(ni):
                print('|',end='',flush=True)
        
        self.counter = dn
        #Reached the end
        if self._disp and self.niter==self.N:
            print('\n')

