#!/usr/bin/env python

'''
Single-layer model
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
    Common interface for three-layer models
    '''
    def __init__(self,grid,forcing,abl):
        self.__grid = grid
        self.__forcing = forcing
        self.__abl = abl
        self.__PHI = None
    
    def solve(self,method='direct',verbose=False):
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
        return self.__grid
    @property
    def abl(self):
        return self.__abl
    @property
    def forcing(self):
        return self.__forcing
    @property
    def PHI(self):
        return self.__PHI
    @PHI.setter
    def PHI(self,value):
        self.__PHI = value

class S1Dmodel(model):
    '''
    Steady one-dimensional gravity wave model
    '''
    def __init__(self,grid,forcing,abl):
        super().__init__(grid,forcing,abl)
        self.PHI = self.PHIvector()

    def format_solution(self,X):
        u1c,v1c = self.expandX(X)
        etac = self.continuity(u1c,v1c)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['etac'] = etac
        result['pc']   = self.PHI*etac
        return result

    def expandX(self,X):
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        return u1, v1

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

    def continuity(self,u1c,v1c):
        '''Return boundar-layer displacement based on given velocity field'''
        return -self.abl.H1/self.abl.U1*u1c

    def Bvector(self):
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
        Find preconditioner M = inv(P) that approximates inv(A)
        (in Smith model, the approximation is exact)
        The matrix vector product Mx is found by solving Py=x
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
        super().__init__(grid,forcing,abl)
        self.PHI = self.PHIvector()

    def format_solution(self,X):
        u1c,v1c,pc = self.expandX(X)
        etac = self.continuity(u1c,v1c,pc)
        result = {}
        result['u1c']  = u1c
        result['v1c']  = v1c
        result['etac'] = etac
        result['pc']   = pc
        return result

    def expandX(self,X):
        N = self.grid.N
        u1 = X[0:N].reshape(self.grid.shape)
        v1 = X[1*N:2*N].reshape(self.grid.shape)
        p  = X[2*N:3*N].reshape(self.grid.shape)
        return u1, v1, p

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

    def continuity(self,u1c,v1c,pc):
        '''Return boundar-layer displacement based on given velocity field'''
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
#                #Hydrostatic solution 
#                if not sigma3==0:
#                    m = np.sign(sigma3)*np.sqrt((k**2+l**2)*(self.abl.N**2/sigma3**2))
#                    PHI[indexk,indexl] += 1j/m*(self.abl.N**2)
        return PHI

    def Mx(self,x):
        '''
        Find preconditioner M = inv(P) that approximates inv(A)
        (in Smith model, the approximation is exact)
        The matrix vector product Mx is found by solving Py=x
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

    Atmospheric boundary-layer model
    '''
    def __init__(self,input='LESbased',**kwargs):
        #input flag indicates how the data is specified
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
        #Default supercritical abl state, corresponding to finWF5
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
        #Default subcritical abl state, corresponding to finSBL_q00
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
        #ABL state based on analytical formulas for u and v (Csanady 1974)
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
        #ABL state based on analytical formulas for u and v (Nieuwstadt 1983)
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
        #ABL state based on analytical formulas for u and v (Nieuwstadt 1983)
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
        #load ABL state from file
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
        U1n = self.U1*np.cos(alpha)+self.V1*np.sin(alpha)
        V1n = self.V1*np.cos(alpha)-self.U1*np.sin(alpha)
        self.__U1 = U1n
        self.__V1 = V1n
        U3n = self.U3*np.cos(alpha)+self.V3*np.sin(alpha)
        V3n = self.V3*np.cos(alpha)-self.U3*np.sin(alpha)
        self.__U3 = U3n
        self.__V3 = V3n

    def saveas(self,filename,info=''):
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
        return self.__us
    @property
    def vs(self):
        return self.__vs
    @property
    def Ms(self):
        return np.sqrt(self.us**2+self.vs**2)
    @property
    def zs(self):
        return self.__zs
    @property
    def zst(self):
        return self.__zst
    @property
    def H1(self):
        return self.__H1
    @property
    def U1(self):
        return self.__U1
    @property
    def V1(self):
        return self.__V1
    @V1.setter
    def V1(self,value):
        self.__V1 = value
    @property
    def S1(self):
        return np.sqrt(self.U1**2 + self.V1**2)
    @property
    def WD1(self):
        return np.arctan(self.V1/self.U1)*180/np.pi
    @property
    def U3(self):
        return self.__U3
    @property
    def V3(self):
        return self.__V3
    @property
    def S3(self):
        return np.sqrt(self.U3**2 + self.V3**2)
    @property
    def WD3(self):
        return np.arctan(self.V3/self.U3)*180/np.pi
    @property
    def C0(self):
        return self.__C0
    @property
    def C1(self):
        return self.__C1
    @property
    def gprime(self):
        return self.__gprime
    @property
    def N(self):
        return self.__N
    @property
    def Fr(self):
        return self.U1/np.sqrt(self.gprime*self.H1)
