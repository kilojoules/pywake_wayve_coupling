#!/usr/bin/env python

'''
Single-layer model developed by Smith 2010
'''

__author__ = "Dries Allaerts"
__date__ = "September 6, 2017"

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

class model(object):
    '''
    Common interface for single-layer models
    '''
    def __init__(self,grid,forcing,abl):
        '''
        Parameters
        ----------
        grid: Grid object (from TLM)
            numerical grid
        forcing: CST/WF object (from TLMForcing)
            perturbing force
        abl: ABL object
            atmospheric state (single-layer)
        '''
        self.__grid = grid
        self.__forcing = forcing
        self.__abl = abl
        self.__PHI = None
    
    def solve(self,method='direct',verbose=False):
        '''
        Solve the single-layer model

        Parameters
        ----------
        method (optional): str
            method used to solve the linear matrix equation
            default: direct solve
        verbose (optional): bool
            flag for printing solver results
            default: False
        '''
        N = self.grid.N
        if method=='direct':
            B = self.Bvector()
            ##########################
            #Solve system of equations
            ##########################
            if verbose:
                print('Start 1D model calculation')
            start = time.time()
            X = self.Mx(B)
            end = time.time()
            if verbose:
                print('Time to calculate 1D model was',end-start,'s')
            ##########################
        else:
            print('Method unknown')
            X = np.zeros((B.shape))
            ##########################

        ##################################
        #Store solution and do inverse fft
        ##################################i
        result = self.format_solution(X)
        if verbose:
            print('Start inverse FFT')
        start = time.time()
        result['u1r'], err_u1r   = self.c2r(result['u1c'],True)
        result['v1r'], err_v1r   = self.c2r(result['v1c'],True)
        result['etar'], err_etar = self.c2r(result['etac'],True)
        result['pr'], err_pr     = self.c2r(result['pc'],True)
        end = time.time()
        if verbose:
            print('Time to compute inverse FFT was',end-start,'s')
            print('Imaginary part of u1r is smaller than',err_u1r)
            print('Imaginary part of v1r is smaller than',err_v1r)
            print('Imaginary part of etar is smaller than',err_etar)
            print('Imaginary part of pr is smaller than',err_pr)
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
    def __init__(self,grid,forcing,abl):
        '''
        Parameters
        ----------
        grid: Grid object (from TLM)
            numerical grid
        forcing: CST/WF object (from TLMForcing)
            perturbing force
        abl: ABL object
            atmospheric state (single-layer)
        '''
        super().__init__(grid,forcing,abl)
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
            keys > u1c,v1c: perturbation velocity
                   etac: inversion displacement
                   pc: pressure perturbation
        '''
        u1c,v1c = self.expandX(X)
        etac = self.continuity(u1c,v1c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
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
        u1, v1: 1d numpy array
            perturbation velocities
        '''
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        return u1, v1

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

    def continuity(self,u1c,v1c):
        '''
        Compute boundar-layer displacement based on given velocity field
        
        Parameters
        ----------
        u1c,v1c: 1d numpy array
            perturbation velocities (Fourier space)

        Returns
        -------
        _: 1d numpy array
            boundary-layer displacement
        '''
        return -self.abl.H1/self.abl.U1*u1c

    def Bvector(self):
        '''
        Compute right-hand side of model equations (B vector)

        Returns
        -------
        _: 1d numpy array
            right-hand side of model equations (Fourier space)
        '''
        #Compute 0th order forcing term (real)
        F0u, F0v = self.forcing.F0(self.abl,self.grid)
        #Convert to fourier space and
        #divide by H1 (SLM solves height-averaged equations)
        Bu = self.r2c(F0u)/self.abl.H1
        Bv = self.r2c(F0v)/self.abl.H1
        #Defunct modes
        Bu[0] = 0.
        Bv[0] = 0.
        return np.concatenate((Bu,Bv))

    def PHIvector(self):
        '''
        Compute complex stratification coefficient Phi

        Returns
        -------
        PHIs: 1d numpy array
            complex stratification coefficient
        '''
        Nx = self.grid.Nx
        PHIs = self.abl.gprime*np.ones((Nx),dtype=np.complex128)
        for index, k in enumerate(self.grid.ks):
            sigma3 = self.abl.U3*k
#            #Non-hydrostatic solution 
#            if (not sigma3==0):
#                if sigma3**2>self.abl.N**2:
#                    m = 1j*np.sqrt(k**2*np.abs(self.abl.N**2/sigma3**2-1))
#                else:
#                    m = np.sign(sigma3)*np.sqrt(k**2*(self.abl.N**2/sigma3**2-1))
#                PHIs[index] += 1j/m*(self.abl.N**2-sigma3**2)
            #Hydrostatic solution 
            if (not sigma3==0):
                m = np.sign(sigma3)*np.sqrt(k**2*self.abl.N**2/sigma3**2)
                PHIs[index] += 1j/m*self.abl.N**2
        return PHIs

    def Mx(self,x):
        '''
        Compute the matrix vector product M*x where preconditioner M = inv(P)
        approximates inv(A) (in Smith model, the approximation is exact)
        The matrix vector product Mx is found by solving Py=x

        Parameters
        ----------
        x: 1d numpy array
            input vectori

        Returns
        -------
        _: 1d numpy array
            matrix vector product M*x
        '''
        N = self.grid.N
        M = np.zeros((2,N),dtype=np.complex128)
        X = x.reshape(2,N)
        for index, k in enumerate(self.grid.ks):    
            P = np.zeros((2,2),dtype=np.complex128)
            sigma1 = self.abl.U1*k
            #u1 equation
            P[0,0] = (-1j*sigma1-self.abl.C0-self.abl.C1
                      +1j*k*self.PHI[index]*self.abl.H1/self.abl.U1)
            #P[0,1] = 0.
            #v1 equation
            #P[1,0] = 0.
            P[1,1] = (-1j*sigma1-self.abl.C0-self.abl.C1)
            M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        M[:,0] = 0.
        return M.reshape(2*N)

class S2Dmodel(model):
    '''
    Steady two-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl):
        '''
        Parameters
        ----------
        grid: Grid object (from TLM)
            numerical grid
        forcing: CST/WF object (from TLMForcing)
            perturbing force
        abl: ABL object
            atmospheric state (single-layer)
        '''
        super().__init__(grid,forcing,abl)
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
            dictionary with 2d numpy arrays
            keys > u1c,v1c: perturbation velocity
                   etac: inversion displacement
                   pc: pressure perturbation
        '''
        u1c,v1c,pc = self.expandX(X)
        etac = self.continuity(u1c,v1c,pc)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['etac'] = etac
        result['pc']   = pc
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
        u1,v1,p: 2d numpy array
            perturbation velocities and pressure
        '''
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        p  = X[2*N:3*N].reshape(self.grid.shape)
        return u1, v1, p

    def c2r(self,cfield,returnErr=False):
        '''
        Perform an inverse Fourier transform

        Parameters
        ----------
        cfield: 2d numpy array
            complex field, assuming
                - Hermitian symmetry
                - zero-wavenumber component at the center of the spectrum
        returnErr (optional): bool
            flag to return the maximum imaginary part of the real field
            default: False

        Returns
        -------
        rfield: 2d numpy array
            inverse Fourier transform of cfield (real part)
        err (optional): float
            maximum imaginary part of rfield (zero if input is Hermitian-symmetric)
        '''
        rfield = np.fft.ifft2(np.fft.ifftshift(cfield*self.grid.N))
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
        rfield: 2d numpy array
            real field

        Returns
        -------
        cfield: 2d numpy array
            Fourier transform of rfield
            (zero-wavenumber at the center of the spectrum)
        '''
        return np.fft.fftshift(np.fft.fft2(rfield))/self.grid.N

    def continuity(self,u1c,v1c,pc):
        '''
        Compute boundar-layer displacement based on given velocity field
        
        Parameters
        ----------
        u1c,v1c,pc: 2d numpy array
            perturbation velocities and pressure (Fourier space)

        Returns
        -------
        eta: 2d numpy array
            boundary-layer displacement
        '''
        ##Using the continuity equation
        #Ks, Ls = np.meshgrid(self.grid.ks,self.grid.ls,indexing='ij')
        #sigma1 = self.abl.U1*Ks+self.abl.V1*Ls
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    eta = -self.abl.H1/sigma1*(Ks*u1c+Ls*v1c)
        ##When sigma1 is exactly zero (this includes the mean mode),
        ##eta is undefined (division by zero). The exact value doesn't matter
        ##because it does not appear in the system of equations (the continuity
        ##equation changes to an incompressiblity condition in the limiting case)
        ##The value of eta is replaced with the limit value p/PHI to make
        ##the eta field continuous. Note that setting eta to zero would cause
        ##broad stripes in the solution field when one of U1,V1 is zero.
        ##There is no issue for values of sigma1 close but not equal to zero
        #eta[sigma1==0.]=pc[sigma1==0.]/self.PHI[sigma1==0.]

        #Using the pressure field:
        #Eta can also be found by the relation p=PHI*eta
        #The result is identical to the displacement found with the continuity
        #equation when the limiting value for sigma1->0 is chosen correctly
        #However, this method is numerically more stable as it only involves a
        #product, whereas the continuity approach can result in the division of
        #two very small numbers (order of 1.0e-21) which is inaccurate
        #The only issue here is when PHI=0., which can occur for gprime=0.
        eta = pc/self.PHI
        return eta

    def Bvector(self):
        '''
        Compute right-hand side of model equations (B vector)

        Returns
        -------
        _: 1d numpy array
            right-hand side of model equations (Fourier space)
        '''
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
        return np.concatenate((Bu,Bv,np.zeros((Nx*Ny,),dtype=np.complex128)))

    def PHIvector(self):
        '''
        Compute complex stratification coefficient Phi

        Returns
        -------
        PHIs: 2d numpy array
            complex stratification coefficient
        '''
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        PHI = self.abl.gprime*np.ones((Nx,Ny),dtype=np.complex128)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                sigma3 = self.abl.U3*k+self.abl.V3*l
#                #Non-hydrostatic solution 
#                if not sigma3==0:
#                    if sigma3**2>self.abl.N**2:
#                        m = 1j*np.sqrt((k**2+l**2)*np.abs(self.abl.N**2/sigma3**2-1))
#                    else:
#                        m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2-1))
#                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2-sigma3**2)
                #Hydrostatic solution 
                if not sigma3==0:
                    m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2))
                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2)
        return PHI

    def Mx(self,x):
        '''
        Compute the matrix vector product M*x where preconditioner M = inv(P)
        approximates inv(A) (in Smith model, the approximation is exact)
        The matrix vector product Mx is found by solving Py=x

        Parameters
        ----------
        x: 1d numpy array
            input vectori

        Returns
        -------
        _: 1d numpy array
            matrix vector product M*x
        '''
        N = self.grid.N
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        M = np.zeros((3,N),dtype=np.complex128)
        X = x.reshape(3,N)
        for indexk, k in enumerate(self.grid.ks):
            for indexl, l in enumerate(self.grid.ls):
                index = indexl + Ny*indexk
                P = np.zeros((3,3),dtype=np.complex128)
                sigma1 = self.abl.U1*k+self.abl.V1*l
                #u1 equation
                P[0,0] = (-1j*sigma1-self.abl.C0-self.abl.C1)
                #P[0,1] = 0.
                P[0,2] = -1j*k
                #v1 equation
                #P[1,0] = 0.
                P[1,1] = (-1j*sigma1-self.abl.C0-self.abl.C1)
                P[1,2] = -1j*l
                #p equation
                #General case
                P[2,0] = self.PHI[indexk,indexl]*self.abl.H1*k
                P[2,1] = self.PHI[indexk,indexl]*self.abl.H1*l
                P[2,2] = sigma1
                #k=l=0
                if k==0 and l == 0:
                    P[2,2] = 1.+0.j

                M[:,index] = scipy.linalg.solve(P,X[:,index])
        #Defunct mode
        defunct_k = [i for i in range(Ny)]
        defunct_l = [i*Ny for i in range(1,Nx)]
        defunctindices = np.array(defunct_k + defunct_l)
        M[:,defunctindices] = 0.+0.j
        return M.reshape(3*N)

class ABL(object):
    '''
    Atmospheric state (single-layer)

    Data structure containing all information about the atmospheric state
    '''
    def __init__(self,input='LESbased',**kwargs):
        '''
        Initialise atmospheric state with one of the following valid methods:
        - default subcritical state
        - default supercritical state
        - based on LES data
        - based on analytic profile with constant eddy viscosity (out-dated)
        - based on analytic profile with quadratic eddy viscosity (out-dated)
        - based on analytic profile with cubic eddy viscosity
        - directly specifying model parameters
        - load from file (written with the ABL.saveas() routine)

        Parameters
        ----------
        input: str
            Name of the method used to specify the atmospheric state
        '''
        assert input in ['default_subcr',
                         'default_supercr',
                         'LESbased',
                         'analytic_constant',
                         'analytic_quadratic',
                         'analytic_cubic',
                         'userdefined',
                         'fromfile'], 'Error: ABL input mode unknown'

        self.__us = None
        self.__vs = None
        self.__zs = None
        self.__zst = None
        
        function = getattr(self,input)
        function(**kwargs)

    def default_supercr(self,**kwargs):
        '''
        Method to specify the atmospheric state as the default implemented
        supercritical state, which corresponds to the atmospheric conditions
        of case S1 of Allaerts and Meyers, J. Fluid Mech. 814, 2017
        '''
        self.__H1 = 1055.0
        self.__U1 = 11.729
        self.__V1 = -0.6873
        self.__U3 = 11.888
        self.__V3 = -1.633
        self.__C0 = 1.55709e-5
        self.__C1 = 1.90501e-7
        self.__gprime = 3.432e-2
        self.__N      = 0.58354e-2

    def default_subcr(self,**kwargs):
        '''
        Method to specify the atmospheric state as the default implemented
        subcritical state, which corresponds to the atmospheric conditions
        of case Q00 of Allaerts and Meyers, Bound. Layer Meteorol. 166(2), 2018
        '''
        self.__H1 = 1060.0
        self.__U1 = 11.508
        self.__V1 = -1.316
        self.__U3 = 11.505
        self.__V3 = -3.4106
        self.__C0 = 4.11828e-5
        self.__C1 = 3.19383e-7
        self.__gprime = 0.1696
        self.__N      = 0.58132e-2

    def LESbased(self,**kwargs):
        '''
        Method to specify the atmospheric state based on LES data

        This routine assumes that the LES data is obtained with SP-Wind, and
        the input parameters are filenames and specific data structures rather
        than general vertical profiles

        Parameters
        ----------
        sim: Simulation object (defined in simulation.py of py4sp package)
            LES simulation meta data structure
        tstart,tend: float
            LES start and end time between which time averages are collected
        ccfilename: str
            Path and filename of BL_tstatcc.dat (or BL_instcc.dat) file
        stfilename: str
            Path and filename of BL_tstatst.dat (or BL_instst.dat) file
        EKfilename: str
            Path and filename of ek_post.dat file
        ENfilename: str
            Path and filename of en_tstatcc.dat file
        '''
        arguments = ['sim','tstart','tend',
                     'ccfilename','stfilename',
                     'EKfilename','ENfilename']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for LESbased ABL definition are missing'

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
        i1 = np.max(np.where(data['zst']<=CIestimate['h1']))-1
        self.__H1 = data['zst'][i1+1]-data['zst'][0]
        self.__U1 = mypy.trapzst(data['u'],data['zst'],0,i1)/self.H1
        self.__V1 = mypy.trapzst(data['v'],data['zst'],0,i1)/self.H1
        tau01 = data['utau']**2
        self.__C0 = 2*tau01/(self.S1*self.H1)
        tau12 = tau[i1+1]
        self.__C1 = 2*tau12/(self.S1*self.H1)
    
        #Other ABL parameters
        self.__gprime = sim.abl.gravity*CIestimate['dth']/sim.Tref
        self.__N  = np.sqrt(sim.abl.gravity*CIestimate['gamma']/sim.Tref)

    def analytic_constant(self,**kwargs):
        '''
        Method to specify the atmospheric state based on analytic profiles with
        a constant eddy viscosity profile. The analytic profiles have been derived
        by Csanady 1974

        This method might be outdated

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
        alpha: float
            Geostrophic wind direction
        viscosity: float
            Eddy viscosity
        utau: float
            Friction velocity
        h: float
            Boundary-layer height
        '''
        arguments = ['dth','fc','N','G','alpha','viscosity','utau','h']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_constant ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = kwargs['h']

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
        self.__U1 = mypy.trapzst(u,zst,0,Nz)/self.H1
        self.__V1 = mypy.trapzst(v,zst,0,Nz)/self.H1
        self.__nu1 = K
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2

    def analytic_quadratic(self,**kwargs):
        '''
        Method to specify the atmospheric state based on analytic profiles with
        a quadratic eddy viscosity profile. The analytic profiles have been derived
        by Nieuwstadt 1983

        This method might be outdated

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
        alpha: float
            Geostrophic wind direction
        kappa: float
            Von Karman constant
        utau: float
            Friction velocity
        h: float
            Boundary-layer height
        '''
        arguments = ['dth','fc','N','G','alpha','kappa','utau','h']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_quadratic ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = kwargs['h']

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
        self.__U1 = mypy.trapzst(u,zst,0,Nz)/self.H1
        self.__V1 = mypy.trapzst(v,zst,0,Nz)/self.H1
        self.__nu1 = mypy.trapzst(kappa*utau*zs*(1-zs/h),zst,0,Nz)/self.H1
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2

    def analytic_cubic(self,**kwargs):
        '''
        Method to specify the atmospheric state based on analytic profiles with
        a cubic eddy viscosity profile. The analytic profiles have been derived by
        Nieuwstadt 1983

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
        alpha: float
            Geostrophic wind direction
        kappa: float
            Von Karman constant
        utau: float
            Friction velocity
        h: float
            Boundary-layer height
        '''
        arguments = ['dth','fc','N','G','alpha','kappa','utau','h']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for analytic_quadratic ABL definition are missing'
        
        self.__gprime = 9.81*kwargs['dth']/288.15
        self.__N  = kwargs['N']
        self.__U3 = kwargs['G']*np.cos(kwargs['alpha'])
        self.__V3 = kwargs['G']*np.sin(kwargs['alpha'])
        self.__H1 = kwargs['h']

        h = kwargs['h']
        kappa = kwargs['kappa']
        utau = kwargs['utau']

        #Velocity deficit vector
        Nz = 100
        zst = np.linspace(0.,h,Nz)
        zs = (zst[0:-1]+zst[1:])/2.0
        C = h*self.fc/kappa/utau
        alpha = 0.5+0.5*np.sqrt(1+4j*C)
        wd = np.zeros((Nz-1),dtype=np.complex128)
        sigma = np.zeros((Nz-1),dtype=np.complex128)
        for k in range(Nz-1):
            wd[k] = (1j*alpha**2*(mpmath.gamma(alpha))**2)/(kappa*C*mpmath.gamma(2*alpha))*(1-zs[k]/h)**(alpha-1)*mpmath.hyp2f1(alpha+1,alpha-1,2*alpha,1-zs[k]/h)
            sigma[k] = (alpha*(mpmath.gamma(alpha))**2)/(mpmath.gamma(2*alpha))*(1-zs[k]/h)**(alpha)*mpmath.hyp2f1(alpha-1,alpha,2*alpha,1-zs[k]/h)

        u = self.U3+np.real(wd)*utau
        v = self.V3+np.imag(wd)*utau
        taux = np.real(sigma)*utau**2
        tauy = np.imag(sigma)*utau**2
        tau = np.sqrt(taux**2+tauy**2)
        #Compute height averaged quantities
        #Layer 1
        self.__U1 = mypy.trapzst(u,zst,0,Nz)/self.H1
        self.__V1 = mypy.trapzst(v,zst,0,Nz)/self.H1
        tau01 = utau**2
        self.__C0 = tau01/self.S1**2

    def userdefined(self,**kwargs):
        '''
        Directly specify atmospheric model parameters

        Parameters
        ----------
        H1: float
            Boundary-layer height
        U1,V1: float
            Boundary-layer velocities in x and y directions
        U3,V3: float
            Free atmosphere velocities in x and y directions
        C0,C1: float
            Friction coefficient at surface and at boundary-layer top
        gprime: float
            Reduced gravity
        N: float
            Brunt Vaisala frequency
        '''
        arguments = ['H1','U1','V1','U3','V3','C0','C1','gprime','N']
        assert all([i in kwargs for i in arguments]), 'Error: some arguments for LESbased ABL definition are missing'

        self.__H1 = kwargs['H1']
        self.__U1 = kwargs['U1']
        self.__V1 = kwargs['V1']
        self.__U3 = kwargs['U3']
        self.__V3 = kwargs['V3']
        self.__C0 = kwargs['C0']
        self.__C1 = kwargs['C1']
        self.__gprime = kwargs['gprime']
        self.__N      = kwargs['N']

    def fromfile(self,**kwargs):
        '''
        Load ABL object from file

        Parameters
        ----------
        filename: str
            File containing the ABL object
        '''
        assert 'filename' in kwargs, 'Error: filename not specified'
        
        #Read from file
        with open(kwargs['filename'],'r') as file:
            for _ in range(7): file.readline()
            self.__H1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__U1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__V1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__U3     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__V3     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__C0     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__C1     = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__gprime = float(file.readline().rstrip('\r\n').split('=')[1])
            self.__N      = float(file.readline().rstrip('\r\n').split('=')[1])
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
        '''
        Rotate coordinate axis over a certain angle

        Parameters
        ----------
        alpha: float
            Angle over which the coordinate axis is to be rotated (in radians)
        '''
        U1n = self.U1*np.cos(alpha)+self.V1*np.sin(alpha)
        V1n = self.V1*np.cos(alpha)-self.U1*np.sin(alpha)
        self.__U1 = U1n
        self.__V1 = V1n
        U3n = self.U3*np.cos(alpha)+self.V3*np.sin(alpha)
        V3n = self.V3*np.cos(alpha)-self.U3*np.sin(alpha)
        self.__U3 = U3n
        self.__V3 = V3n

    def saveas(self,filename,info=''):
        '''
        Save ABL object to file

        Parameters
        ----------
        filename: str
            Name of destination file
        info (optional): str
            String with information about the ABL object
            Default: Empty string
        '''
        with open(filename,'w') as file:
            file.write('%%%%%%%%%%%%%%\n')
            file.write('SLM ABL object\n')
            file.write('%%%%%%%%%%%%%%\n')
            file.write(info+'\n')
            file.write('\n')
            file.write('1. Scalar data\n')
            file.write('--------------\n')
            file.write('H1     ='+'{:17.10g}'.format(self.H1)+'\n')
            file.write('U1     ='+'{:17.10g}'.format(self.U1)+'\n')
            file.write('V1     ='+'{:17.10g}'.format(self.V1)+'\n')
            file.write('U3     ='+'{:17.10g}'.format(self.U3)+'\n')
            file.write('V3     ='+'{:17.10g}'.format(self.V3)+'\n')
            file.write('C0     ='+'{:17.10g}'.format(self.C0)+'\n')
            file.write('C1     ='+'{:17.10g}'.format(self.C1)+'\n')
            file.write('gprime ='+'{:17.10g}'.format(self.gprime)+'\n')
            file.write('N      ='+'{:17.10g}'.format(self.N)+'\n')

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
    def zs(self):
        '''Height corresponding to the vertical profiles (at cell centers)'''
        return self.__zs
    @property
    def zst(self):
        '''Height corresponding to the vertical profiles (at cell faces)'''
        return self.__zst
    @property
    def H1(self):
        '''Boundary-layer height'''
        return self.__H1
    @property
    def U1(self):
        '''Boundary-layer velocity in dimension 0'''
        return self.__U1
    @property
    def V1(self):
        '''Boundary-layer velocity in dimension 1'''
        return self.__V1
    @V1.setter
    def V1(self,value):
        self.__V1 = value
    @property
    def S1(self):
        '''Boundary-layer velocity magnitude'''
        return np.sqrt(self.U1**2 + self.V1**2)
    @property
    def WD1(self):
        '''
        Wind direction in the boundary layer (degrees)
    
        Bug: np.arctan only recognises angles between -90 and +90
        Better would be to return np.arctan2(self.V1,self.U1)*180/np.pi
        Even better is to return the actual wind direction:
            return 180. + np.arctan2(self.U1,self.V1)*180/np.pi
        '''
        return np.arctan(self.V1/self.U1)*180/np.pi
    @property
    def U3(self):
        '''Velocity in the free atmosphere in dimension 0'''
        return self.__U3
    @property
    def V3(self):
        '''Velocity in the free atmosphere in dimension 1'''
        return self.__V3
    @property
    def S3(self):
        '''Velocity magnitude in the free atmosphere'''
        return np.sqrt(self.U3**2 + self.V3**2)
    @property
    def WD3(self):
        '''
        Wind direction in the free atmosphere (degrees)
    
        Bug: see WD1
        '''
        return np.arctan(self.V3/self.U3)*180/np.pi
    @property
    def C0(self):
        '''Friction coefficient at the surface'''
        return self.__C0
    @property
    def C1(self):
        '''Friction coefficient at the boundary-layer top'''
        return self.__C1
    @property
    def gprime(self):
        '''Reduced gravity'''
        return self.__gprime
    @property
    def N(self):
        '''Brunt Vaisala frequency'''
        return self.__N
    @property
    def Fr(self):
        '''Froude number'''
        return self.U1/np.sqrt(self.gprime*self.H1)
