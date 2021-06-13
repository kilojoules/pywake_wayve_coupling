#!/usr/bin/env python

'''
Some additional function
'''

__author__ = "Luca Lanzilao, Dries Allaerts"
__date__   = "August 22, 2017"

import numpy as np
from numba import njit
import numba
import mpmath

def Cg_cubic(hstar,z0_h,kappa):
    '''
    Geostrophic drag Cg = utau/G as a function of hstar = h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    z0_h: float
        Non-dimensional surface roughness length
        z0_h = z0/h
    kappa: float
        Von Karman constant

    Returns
    -------
    Cg: float
        Geostrophic drag
        Cg = utau/G
    '''
    F1 = F1_cubic(hstar,kappa)
    F2 = F2_cubic(hstar,kappa)
    Cg = (F2**2+(1./kappa*np.log(1./(hstar*z0_h))-F1)**2)**(-1/2)
    return Cg

def alpha_cubic(hstar,z0_h,kappa):
    '''
    Geostrophic wind direction alpha as a function of hstar = h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    z0_h: float
        Non-dimensional surface roughness length
        z0_h = z0/h
    kappa: float
        Von Karman constant

    Returns
    -------
    alpha: float
        Geostrophic wind direction
    '''
    Cg = Cg_cubic(hstar,z0_h,kappa)
    F2 = F2_cubic(hstar,kappa)
    alpha = np.arcsin(-Cg*F2)
    return alpha

def F1_cubic(hstar,kappa):
    '''
    F1 function corresponding to a cubic viscosity profile (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    kappa: float
        Von Karman constant

    Returns
    -------
    F1: float
        Value of F1 function
    '''
    C = hstar/kappa
    alpha = 0.5+0.5*np.sqrt(1+4j*C)
    F1 = np.zeros((1),dtype=np.float64)
    F1[0] = 1./kappa*(-np.log(hstar)+
                      mpmath.re( mpmath.digamma(alpha+1)+
                                 mpmath.digamma(alpha-1)-
                                 2*mpmath.digamma(1.0) ) )
    return np.asscalar(F1)

def F2_cubic(hstar,kappa):
    '''
    F2 function corresponding to a cubic viscosity profile (Nieuwstadt 1983)

    Parameters
    ----------
    hstar: float
        Non-dimensional boundary-layer height
        hstar = h*fc/utau
    kappa: float
        Von Karman constant

    Returns
    -------
    F2: float
        Value of F2 function
    '''
    C = hstar/kappa
    alpha = 0.5+0.5*np.sqrt(1+4j*C)
    F2 = np.zeros((1),dtype=np.float64)
    F2[0] = 1./kappa*(mpmath.im( mpmath.digamma(alpha+1)+
                                 mpmath.digamma(alpha-1)-
                                 2*mpmath.digamma(1.0) ) )
    return np.asscalar(F2)

@njit(parallel=False)
def evaluate_A(U1,V1,U2,V2,S1,dU12,dV12,dS12,H1,H2,C0,C1,nu1,nu2,fc,Nx2,Ny,N2,
               u1,v1,u2,v2,p1,p2,ks2,ls,PHI,F1u,F1v):
    '''
    Evaluate the matrix product A*x in Numba syntax

    Parameters
    ----------
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    S1: float
        Height-averaged velocity magnitude in the wind-farm layer
    dU12: float
        Velocity difference between the wind-farm and upper layer in dimension 0
    dV12: float
        Velocity difference between the wind-farm and upper layer in dimension 1
    dS12: float
        Magnitude of velocity difference vector between the wind-farm and
        upper layer
    H1: float
        Height of the wind-farm layer
    H2: float
        Height of the upper layer
    C0: float
        Friction coefficient at the surface
    C1: float
        Friction coefficient at the interface between wind-farm and upper layer
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    fc: float
        Coriolis parameter
    Nx: float
        Number of grid points in dimension 0
    Ny: float
        Number of grid points in dimension 1
    N: float
        Total number of grid points
    u1,v1,u2,v2,p1,p2: 1d numpy array
        Three-layer model solution of current LGMRES iteration
    ks,ls: 1d numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    PHI: 2d numpy array
        Complex stratification coefficient
    F1u,F1v: 2d numpy array
        First-order term of forcing term taylor expansion

    Returns
    -------
    Axu1,Axv1,Axu2,Axv2,Axp1,Axp2: 1d numpy array
        Matrix product A*x
    '''
    #Initialise components of the matrix vector product
    Axu1 = np.zeros((N2),dtype=np.complex128)
    Axv1 = np.zeros((N2),dtype=np.complex128)
    Axu2 = np.zeros((N2),dtype=np.complex128)
    Axv2 = np.zeros((N2),dtype=np.complex128)
    Axp1 = np.zeros((N2),dtype=np.complex128)
    Axp2 = np.zeros((N2),dtype=np.complex128)
    
    for index in numba.prange(N2):
        sigma1 = U1*ks2[index]+V1*ls[index]
        sigma2 = U2*ks2[index]+V2*ls[index]
        #Regular entries
        #u1 equation
        Axu1[index] = ( - 1j*sigma1*u1[index]
                        - 1j*ks2[index]*(p1[index]+p2[index])
                        + fc*v1[index] 
                        - C0*(S1**2+U1**2)/(S1*H1)*u1[index]
                        - C0*(U1*V1)/(S1*H1)*v1[index]
                        - C1*(dS12**2+dU12**2)/(dS12*H1)*(u1[index]-u2[index])
                        - C1*(dU12*dV12)/(dS12*H1)*(v1[index]-v2[index])
                        - nu1*(ks2[index]**2+ls[index]**2)*u1[index] 
                        - F1u[index]/H1 )
        
        #v1 equation
        Axv1[index] = ( - 1j*sigma1*v1[index]
                        - 1j*ls[index]*(p1[index]+p2[index])
                        - fc*u1[index]
                        - C0*(S1**2+V1**2)/(S1*H1)*v1[index]
                        - C0*(U1*V1)/(S1*H1)*u1[index]
                        - C1*(dS12**2+dV12**2)/(dS12*H1)*(v1[index]-v2[index])
                        - C1*(dU12*dV12)/(dS12*H1)*(u1[index]-u2[index])
                        - nu1*(ks2[index]**2+ls[index]**2)*v1[index] 
                        - F1v[index]/H1 )
        #u2 equation
        Axu2[index] = ( - 1j*sigma2*u2[index]
                        - 1j*ks2[index]*(p1[index]+p2[index])
                        + fc*v2[index]
                        + C1*(dS12**2+dU12**2)/(dS12*H2)*(u1[index]-u2[index])
                        + C1*(dU12*dV12)/(dS12*H2)*(v1[index]-v2[index])
                        - nu2*(ks2[index]**2+ls[index]**2)*u2[index] )
        #v2 equation
        Axv2[index] = ( - 1j*sigma2*v2[index]
                        - 1j*ls[index]*(p1[index]+p2[index])
                        - fc*u2[index]
                        + C1*(dS12**2+dV12**2)/(dS12*H2)*(v1[index]-v2[index])
                        + C1*(dU12*dV12)/(dS12*H2)*(u1[index]-u2[index])
                        - nu2*(ks2[index]**2+ls[index]**2)*v2[index] )
        #p1 equation
        #General case
        Axp1[index] = ( + PHI[index]*H1*ks2[index]*u1[index]
                        + PHI[index]*H1*ls[index]*v1[index]
                        + sigma1*p1[index] )
        #p2 equation
        #General case
        Axp2[index] = ( + PHI[index]*H2*ks2[index]*u2[index]
                        + PHI[index]*H2*ls[index]*v2[index]
                        + sigma2*p2[index] )
    
    #p1,2 equation: k=l=0 
    #the zero mode for k and l is in column Ny/2 at row 0. Infact, now the zero
    #mode is located in row 0
    indices_klzero = int(Ny/2)
    Axp1[indices_klzero] = p1[indices_klzero]
    Axp2[indices_klzero] = p2[indices_klzero]
              
    return Axu1,Axv1,Axu2,Axv2,Axp1,Axp2

@njit(parallel=True)
def evaluate_M(ks2,ls,Ny,Nx2,U1,V1,U2,V2,C0,C1,S1,H1,dS12,dU12,nu1,nu2,dV12,fc,H2,PHI,N2,X):
    '''
    Evaluate three layer model preconditioner in Numba syntax

    Parameters
    ----------
    ks,ls: 1d numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    Nx: float
        Number of grid points in dimension 0
    Ny: float
        Number of grid points in dimension 1  
    U1: float
        Height-averaged velocity in the wind-farm layer in dimension 0
    V1: float
        Height-averaged velocity in the wind-farm layer in dimension 1
    U2: float
        Height-averaged velocity in the upper layer in dimension 0
    V2: float
        Height-averaged velocity in the upper layer in dimension 1
    C0: float
        Friction coefficient at the surface
    C1: float
        Friction coefficient at the interface between wind-farm and upper layer
    S1: float
        Height-averaged velocity magnitude in the wind-farm layer
    H1: float
        Height of the wind-farm layer
    dS12: float
        Magnitude of velocity difference vector between the wind-farm and
        upper layer
    dU12: float
        Velocity difference between the wind-farm and upper layer in dimension 0
    nu1: float
        Height-averaged turbulent viscosity in the wind-farm layer
    nu2: float
        Height-averaged turbulent viscosity in the upper layer
    dV12: float
        Velocity difference between the wind-farm and upper layer in dimension 1
    fc: float
        Coriolis parameter
    H2: float
        Height of the upper layer
    PHI: 2d numpy array
        Complex stratification coefficient
    N: float
        Total number of grid points
    X: 1d numpy array
        Three-layer model solution of current LGMRES iteration
        
    Returns
    -------
    M: 2d numpy array
        Preconditioner
    '''
    M = np.zeros((6,N2),dtype=np.complex128)
    for indexk in numba.prange(ks2.size):
        for indexl in numba.prange(ls.size):
            index = indexl + Ny*indexk
            P = np.zeros((6,6),dtype=np.complex128)
            sigma1 = U1*ks2[indexk]+V1*ls[indexl]
            sigma2 = U2*ks2[indexk]+V2*ls[indexl]
            #u1 equation
            P[0,0] = -1j*sigma1-C0*(S1**2+U1**2)/(S1*H1)-C1*(dS12**2+dU12**2)/(dS12*H1)-nu1*(ks2[indexk]**2+ls[indexl]**2) 
            P[0,1] = +fc-C0*U1*V1/(S1*H1)-C1*dU12*dV12/(dS12*H1) 
            P[0,2] = +C1*(dS12**2+dU12**2)/(dS12*H1)
            P[0,3] = +C1*(dU12*dV12)/(dS12*H1)
            P[0,4] = -1j*ks2[indexk]
            P[0,5] = -1j*ks2[indexk]
            #v1 equation
            P[1,0] = -fc-C0*U1*V1/(S1*H1)-C1*dU12*dV12/(dS12*H1) 
            P[1,1] = -1j*sigma1-C0*(S1**2+V1**2)/(S1*H1)-C1*(dS12**2+dV12**2)/(dS12*H1)-nu1*(ks2[indexk]**2+ls[indexl]**2)
            P[1,2] = +C1*(dU12*dV12)/(dS12*H1)
            P[1,3] = +C1*(dS12**2+dV12**2)/(dS12*H1)
            P[1,4] = -1j*ls[indexl]
            P[1,5] = -1j*ls[indexl]
            #u2 equation
            P[2,0] = +C1*(dS12**2+dU12**2)/(dS12*H2)
            P[2,1] = +C1*(dU12*dV12)/(dS12*H2)
            P[2,2] = -1j*sigma2-C1*(dS12**2+dU12**2)/(dS12*H2)-nu2*(ks2[indexk]**2+ls[indexl]**2) 
            P[2,3] = +fc-C1*(dU12*dV12)/(dS12*H2)
            P[2,4] = -1j*ks2[indexk]
            P[2,5] = -1j*ks2[indexk]
            #v2 equation
            P[3,0] = +C1*(dU12*dV12)/(dS12*H2)
            P[3,1] = +C1*(dS12**2+dV12**2)/(dS12*H2) 
            P[3,2] = -fc-C1*(dU12*dV12)/(dS12*H2)
            P[3,3] = -1j*sigma2-C1*(dS12**2+dV12**2)/(dS12*H2)-nu2*(ks2[indexk]**2+ls[indexl]**2)
            P[3,4] = -1j*ls[indexl]
            P[3,5] = -1j*ls[indexl]
            #p1 equation
            #General case
            P[4,0] = PHI[indexk,indexl]*H1*ks2[indexk]
            P[4,1] = PHI[indexk,indexl]*H1*ls[indexl]
            P[4,4] = sigma1
            #p2 equation
            #General case
            P[5,2] = PHI[indexk,indexl]*H2*ks2[indexk]
            P[5,3] = PHI[indexk,indexl]*H2*ls[indexl]
            P[5,5] = sigma2
            #k=l=0
            if ks2[indexk]==0 and ls[indexl] == 0:
                P[4,4] = 1.+0.j
                P[5,5] = 1.+0.j
            #PHI=0
            if PHI[indexk,indexl]==0.:
                P[4,4] = 1.+0.j
                P[5,5] = 1.+0.j
            
            #M[:,index] = np.linalg.solve(P,X[:,index])        #Use on HPC with parallel=False
            M[:,index] = np.dot(np.linalg.inv(P),X[:,index])   #Use on PC             
    return M

@njit(parallel=False)
def evaluate_PHI(gprime,shape2,ks2,ls,U3,V3,N,FA_condition):  
    '''
    Compute complex stratification coefficient Phi in Numba synatx

    Parameters
    ----------
    gprime: float
        Reduced gravity
    shape: 1d numpy array
        Dimensions of the numerical domain
    ks,ls: 1d numpy array
        Array with wave numbers in dimension 0 (Fourier space)
        Array with wave numbers in dimension 1 (Fourier space)
    U3: float
        Velocity in the free atmosphere in dimension 0
    V3: float
        Velocity in the free atmosphere in dimension 1
    N: float
        Total number of grid points
    FA_condition: bool
        free atmosphere condition (hydrostatic, non-hydrostatic)
        
    Returns
    -------
    PHI: 2d numpy array
        complex stratification coefficient
    '''
    PHI = gprime*np.ones(shape2,dtype=np.complex128) 
    if FA_condition == 'hydrostatic':      
        for indexk in numba.prange(ks2.size):
            for indexl in numba.prange(ls.size):
                sigma3 = U3*ks2[indexk] + V3*ls[indexl]
                if not sigma3==0:
                    m = np.sign(sigma3)*np.sqrt((ks2[indexk]**2+ls[indexl]**2)*(N**2/sigma3**2))
                    PHI[indexk,indexl] += 1j/m*(N**2)
    if FA_condition == 'non-hydrostatic':      
        for indexk in numba.prange(ks2.size):
            for indexl in numba.prange(ls.size):
                sigma3 = U3*ks2[indexk] + V3*ls[indexl]
                if not sigma3==0:
                    if sigma3**2>N**2:
                        m = 1j*np.sqrt((ks2[indexk]**2+ls[indexl]**2)*np.abs(N**2/sigma3**2-1))
                    else:
                        m = np.sign(sigma3)*np.sqrt((ks2[indexk]**2+ls[indexl]**2)*(N**2/sigma3**2-1))
                    PHI[indexk,indexl] += 1j/m*(N**2-sigma3**2)

        
    return PHI

@njit(parallel=False)
def evaluate_F0(e_str,F0u,F0v,Ct,rotorarea,St,row,col,value):
    '''
    Compute zero-th order wind-farm force on specified grid in Numba synatx

    Parameters
    ----------
    e_str: 1d numpy array
        Unit vector along the wind direction
    F0u,F0v: 2d numpy array
        zero-th order wind-farm force in x and y (same shape as grid)
    Ct: Float
        Turbine thrust coefficient
    rotorarea: float
        rotor swept area
    St: float
        turbine inflow velocity
    row: 1d numpy array
        row index of wind turbine footprint
    col: 1d numpy array
        column index of wind turbine footprint
    val: 1d numpy array
        value of wind turbine footprint
        
    Returns
    -------
    F0u,F0v: 2d numpy array
        zero-th order wind-farm force in x and y (same shape as grid)
    '''
    for i in numba.prange(len(value)):
            F0u[row[i],col[i]] = F0u[row[i],col[i]] + (0.5 * Ct * rotorarea *
                    St**2 * e_str[0] * value[i] )
            F0v[row[i],col[i]] = F0v[row[i],col[i]] + (0.5 * Ct * rotorarea *
                    St**2 * e_str[1] * value[i] )
    return F0u,F0v

@njit(parallel=False)
def evaluate_F1(e_str,E_str,F1u,F1v,Ct,rotorarea,St,Stjac0,Stjac1,row,col,value,u1inf,v1inf):
    '''
    Compute first-th order wind-farm force on specified grid in Numba synatx

    Parameters
    ----------
    e_str: 1d numpy array
        Unit vector along the wind direction
    E_str: 2d numpy array
        Jacobian of the streamwise unit vector
    F1u,F1v: 2d numpy array
        first-th order wind-farm force in x and y (same shape as grid)
    Ct: Float
        Turbine thrust coefficient
    rotorarea: float
        rotor swept area
    St: float
        turbine inflow velocity
    Stjac0: float
        derivative of turbine inflow velocity along dimension 0
    Stjac1: float
        derivative of turbine inflow velocity along dimension 1
    row: 1d numpy array
        row index of wind turbine footprint
    col: 1d numpy array
        column index of wind turbine footprint
    val: 1d numpy array
        value of wind turbine footprint
    u1inf,v1inf: float
        velocity 10D upstream of first turbine
        
    Returns
    -------
    F1u,F1v: 2d numpy array
        first-th order wind-farm force in x and y (same shape as grid)
    '''
    for i in numba.prange(len(value)):
            F1u[row[i],col[i]] = F1u[row[i],col[i]] + (0.5 * Ct * rotorarea *
                        ( 2*St*Stjac0*e_str[0] +
                         St**2*E_str[0,0] ) * u1inf +
                   (0.5 * Ct * rotorarea *
                        ( 2*St*Stjac1*e_str[0] +
                            St**2*E_str[0,1] ) * v1inf ) ) * value[i]
            F1v[row[i],col[i]] = F1v[row[i],col[i]] + (0.5 * Ct * rotorarea *
                        ( 2*St*Stjac0*e_str[1] +
                            St**2*E_str[1,0] ) * u1inf +
                   (0.5 * Ct * rotorarea *
                        ( 2*St*Stjac1*e_str[1] +
                            St**2*E_str[1,1] ) * v1inf ) ) * value[i]
                
    return F1u,F1v

@njit(parallel=False)
def evaluate_P0(P0,Cp,rotorarea,St,row,col,value):
    '''
    Compute zero-th order wind-farm power on specified grid in Numba synatx

    Parameters
    ----------
    P0: 2d numpy array
        zero-th order wind-farm power (same shape as grid)
    Cp: Float
        Turbine power coefficient
    rotorarea: float
        rotor swept area
    St: float
        turbine inflow velocity
    row: 1d numpy array
        row index of wind turbine footprint
    col: 1d numpy array
        column index of wind turbine footprint
    val: 1d numpy array
        value of wind turbine footprint
        
    Returns
    -------
    P0: 2d numpy array
        zero-th order wind-farm power (same shape as grid)
    '''
    for i in numba.prange(len(value)):
        P0[row[i],col[i]] = P0[row[i],col[i]] + 0.5 * Cp * rotorarea * St**3 * value[i] 
    return P0
    
@njit (parallel=False)
def evaluate_P1(P1,Cp,rotorarea,St,Stjac0,Stjac1,row,col,value,u1inf,v1inf):
    '''
    Compute first-th order wind-farm power on specified grid in Numba synatx

    Parameters
    ----------
    P1: 2d numpy array
        first-th order wind-farm power (same shape as grid)
    Cp: Float
        Turbine power coefficient
    rotorarea: float
        rotor swept area
    St: float
        turbine inflow velocity
    Stjac0: float
        derivative of turbine inflow velocity along dimension 0
    Stjac1: float
        derivative of turbine inflow velocity along dimension 1
    row: 1d numpy array
        row index of wind turbine footprint
    col: 1d numpy array
        column index of wind turbine footprint
    val: 1d numpy array
        value of wind turbine footprint
    u1inf,v1inf: float
        velocity 10D upstream of first turbine
        
    Returns
    -------
    P1: 2d numpy array
        first-th order wind-farm power (same shape as grid)
    '''
    for i in numba.prange(len(value)):
        P1[row[i],col[i]] = P1[row[i],col[i]] + (0.5 * Cp * rotorarea * 3* St**2 * Stjac0 * u1inf * value[i] +
                            0.5 * Cp * rotorarea * 3* St**2 * Stjac1 * v1inf * value[i] ) 
    return P1

def plot_residual(counter,method):
    '''
    Plot residual as function of iterative solver iterations
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 12
    plt.figure()
    indices_orig = np.array([1,2])
    if method == 'LGMRES':
        residual_orig = np.array([9.58e-5,5.02e-17])
    if method == 'GCROT(m,k)':
        residual_orig = np.array([9.58e-5,2.33e-17])        
    indices = np.nonzero(counter.residu)[0]
    residual = counter.residu[indices]
    plt.semilogy(indices_orig,residual_orig,linestyle='--',color='sienna',marker='s', 
                 markerfacecolor='white',markeredgecolor='sienna',markeredgewidth=2.0,
                 linewidth=2, markersize=7,zorder=1)
    plt.semilogy(indices,residual,linestyle='--',color='darkblue',marker='s', 
                 markerfacecolor='white',markeredgecolor='darkblue',markeredgewidth=2.0,
                 linewidth=2, markersize=7,zorder=1)
    plt.axhline(y=1e-10,color='magenta',linestyle='--')
    plt.rc('xtick',labelsize=14) 
    plt.rc('ytick',labelsize=14)
    plt.legend(('Non-Hermitian','Hermitian','Threshold'),fontsize=12)
    plt.xlabel(method+' iteration',fontsize=14)
    plt.ylabel('Residual',fontsize=14)
    plt.rc('xtick',labelsize=14) 
    plt.rc('ytick',labelsize=14)
    plt.xlim([0.9,5])
    #plt.xticks([0,3,6,9,12,15])
    plt.savefig(method+'_iteration.png',format='png',bbox_inches='tight',dpi=300)

















