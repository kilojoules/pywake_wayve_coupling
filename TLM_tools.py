#!/usr/bin/env python

'''
Some additional function
'''

__author__ = "Dries Allaerts"
__date__   = "August 22, 2017"

import numpy as np
import os
import mpmath

def Cg_cubic(hstar,z0_h,kappa):
    '''
    Cg as a function of hstar=h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt
    '''
    F1 = F1_cubic(hstar,kappa)
    F2 = F2_cubic(hstar,kappa)
    Cg = (F2**2+(1./kappa*np.log(1./(hstar*z0_h))-F1)**2)**(-1/2)
    return Cg

def alpha_cubic(hstar,z0_h,kappa):
    '''
    alpha as a function of hstar=h*fc/utau and z0/h
    corresponding to cubic viscosity profiles (Nieuwstadt
    '''
    Cg = Cg_cubic(hstar,z0_h,kappa)
    F2 = F2_cubic(hstar,kappa)
    alpha = np.arcsin(-Cg*F2)
    return alpha

def F1_cubic(hstar,kappa):
    '''
    F1 function corresponding to a cubic viscosity profile (Nieuwstadt)
    hstar = h*fc/utau
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
    F2 function corresponding to a cubic viscosity profile (Nieuwstadt)
    hstar = h*fc/utau
    '''
    C = hstar/kappa
    alpha = 0.5+0.5*np.sqrt(1+4j*C)
    F2 = np.zeros((1),dtype=np.float64)
    F2[0] = 1./kappa*(mpmath.im( mpmath.digamma(alpha+1)+
                                 mpmath.digamma(alpha-1)-
                                 2*mpmath.digamma(1.0) ) )
    return np.asscalar(F2)

