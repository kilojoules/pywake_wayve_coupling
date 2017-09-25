#!/usr/bin/env python

'''
Wake models
'''

__author__ = "Dries Allaerts"
__date__ = "August 8, 2017"

import numpy as np
import scipy.linalg
from py4sp.mypy import step
from py4sp.mypy import pulse

def gauss(turbines,abl):
    Uinf = abl.U1
    Vinf = abl.V1
    Sinf = abl.S1
    #Compute velocities
    Nt = len(turbines)                  #Number of turbines
    e_str = e_streamwise(Uinf,Vinf)     #Unit vector along the wind direction
    TI = gauss_TI(turbines,abl)
    A = gauss_A(turbines,abl,TI)
    B = Sinf*np.ones((Nt))
    S = scipy.linalg.solve(A,B)
    #Compute forces
    Ft = np.zeros((Nt,2),dtype=np.float64)
    for index,turb in enumerate(turbines):
        Ft[index,0] = 0.5* turb.Ct * turb.rotorarea * S[index]**2 * e_str[0]
        Ft[index,1] = 0.5* turb.Ct * turb.rotorarea * S[index]**2 * e_str[1]
    return Ft

def gauss_jac(turbines,abl):
    Uinf = abl.U1
    Vinf = abl.V1
    Sinf = abl.S1
    Nt = len(turbines)                  #Number of turbines
    Ftjac = np.zeros((Nt,2,2))
    #Find dSdu,dSdv
    TI = gauss_TI(turbines,abl)
    A = gauss_A(turbines,abl,TI)
    B = Sinf*np.ones((Nt))
    S = scipy.linalg.solve(A,B)
    dAdu,dAdv = gauss_Ajac(turbines,abl,TI)
    dSinfdu = Uinf/Sinf*np.ones((Nt))
    dSinfdv = Vinf/Sinf*np.ones((Nt))
    dSdu = scipy.linalg.solve(A,dSinfdu-np.dot(dAdu,S))
    dSdv = scipy.linalg.solve(A,dSinfdv-np.dot(dAdv,S))
    #Unit vectors and their derivatives
    e_str  = e_streamwise(Uinf,Vinf)
    E_str  = e_str_jac(Uinf,Vinf)
    for index,turb in enumerate(turbines):
        Ftjac[index,0,0] = (0.5* turb.Ct * turb.rotorarea*
                            ( 2*S[index]*dSdu[index]*e_str[0]
                            +S[index]**2*E_str[0,0]) ) #Fudu
        Ftjac[index,0,1] = (0.5* turb.Ct * turb.rotorarea*
                            ( 2*S[index]*dSdv[index]*e_str[0]
                            +S[index]**2*E_str[0,1]) ) #Fudv
        Ftjac[index,1,0] = (0.5* turb.Ct * turb.rotorarea*
                            ( 2*S[index]*dSdu[index]*e_str[1]
                            +S[index]**2*E_str[1,0]) ) #Fvdu
        Ftjac[index,1,1] = (0.5* turb.Ct * turb.rotorarea*
                            ( 2*S[index]*dSdv[index]*e_str[1]
                            +S[index]**2*E_str[1,1]) ) #Fvdv
    return Ftjac

def gauss_alternative(turbines,Uinf,Vinf,kwake):
#Depreciated
    Nt = len(turbines)                  #Number of turbines
    Nq = 16                             #Number of quadrature points
    Sinf = np.sqrt(Uinf**2+Vinf**2)     #Wind speed
    e_str = np.array([Uinf,Vinf])/Sinf  #Unit vector along the wind direction
    e_span = np.array([-Vinf,Uinf])/Sinf #Unit vector in cross wind direction
    #Sort turbines along wind direction
    xs = np.array([turbines[i].x for i in range(Nt)])
    ys = np.array([turbines[i].y for i in range(Nt)])
    coordinates = np.concatenate([xs,ys]).reshape(Nt,2,order='F')
    dist = np.dot(coordinates,e_str)
    order = np.argsort(dist)
    turbines_sort = [turbines[i] for i in order]
    #Compute velocities
    S = np.zeros((Nt))
    S[0] = Sinf
    for i in range(1,Nt):
        turbi = turbines_sort[i]
        wake_deficit = 0.
        for k in range(0,i):
            turbk = turbines_sort[k]
            #Loop over quadrature points
            for n in range(Nq):
                wn  = 1.0/Nq
                phi = 2*np.pi*n/Nq
                r   = np.sqrt((3+(1-2*(n%2))*np.sqrt(3))/6)*turbi.D/2.0
                xn  = turbi.x + r*np.cos(phi)*e_span[0]
                yn  = turbi.y + r*np.cos(phi)*e_span[1]
                zn  = r*np.sin(phi) #Assuming all turbines have same zh
                #Vector from turbine K to point n on turbine I
                KI = np.array([xn-turbk.x,yn-turbk.y])
                #Streamwise distance between turbine K and turbine I along e_str
                #Positive if I is downstream of K 
                #Negative if I is upstream of K
                delta_str = np.dot(KI,e_str)
                #Spanwise distance between turbine K and turbine I along e_span
                #Has a sign but wake model is axisymmetric
                delta_span = np.dot(KI,e_span)
                #Compute wakedeficit at turbine I due to turbine K
                if delta_str>turbk.D:
                    wake_deficit += S[k]*wn*(1-gauss_wakedeficit(delta_str,
                                                                 delta_span,zn,
                                                                 turbk.D,turbk.Ct,
                                                                 kwake) )
                #Ghost turbine?
        S[i] = Sinf - wake_deficit
    #Unsort turbines
    inv_order = np.argsort(order)
    return S[inv_order]

def gauss_TI(turbines,abl):
    Nt = len(turbines)                  #Number of turbines
    Uinf = abl.U1
    Vinf = abl.V1
    e_str = e_streamwise(Uinf,Vinf)     #Unit vector along the wind direction
    e_span = e_spanwise(Uinf,Vinf)      #Unit vector in cross wind direction
    TI = np.zeros((Nt))
    #Sort turbines along wind direction
    xs = np.array([turbines[i].x for i in range(Nt)])
    ys = np.array([turbines[i].y for i in range(Nt)])
    coordinates = np.concatenate([xs,ys]).reshape(Nt,2,order='F')
    dist = np.dot(coordinates,e_str)
    order = np.argsort(dist)
    turbines_sort = [turbines[i] for i in order]
    #Compute turbulent intensities
    TI[0] = abl.TI
    for i in range(1,Nt):
        turbi = turbines_sort[i]
        TIadded = np.zeros((i))
        #Compute added streamwise turbulent intensity induced by turbine k
        #at turbine i
        for k in range(0,i):
            kwake = abl.kwake(TI[k])
            turbk = turbines_sort[k]
            #Vector from turbine K to point n on turbine I
            KI = np.array([turbi.x-turbk.x,turbi.y-turbk.y])
            #Streamwise distance between turbine K and turbine I along e_str
            #Positive if I is downstream of K 
            #Negative if I is upstream of K
            delta_str = np.dot(KI,e_str)
            #Spanwise distance between turbine K and turbine I along e_span
            #Has a sign but wake model is axisymmetric
            delta_span = np.dot(KI,e_span)
            #Turbine k lies upstream of Turbine i
            if delta_str>0:# and delta_str<15*turbk.D:
                beta = 0.5*(1+np.sqrt(1-turbk.Ct))/np.sqrt(1-turbk.Ct)
                eps = 0.2*np.sqrt(beta)
                sigma = kwake*delta_str+turbk.D*eps
                rwake = 2*sigma
                rdisk = turbi.D/2.0
                #TI wake of k intersects with rotor i
                if np.abs(delta_span)>=(rwake+rdisk):
                    Aw = 0. #No overlap
                elif np.abs(delta_span)>=max([rwake,rdisk]):
                    x = (delta_span**2-rdisk**2+rwake**2)/(2*np.abs(delta_span))
                    Aw = ( area_circle_segment(rwake,x)
                          +area_circle_segment(rdisk,np.abs(delta_span)-x) )
                elif np.abs(delta_span)>=np.abs(rwake-rdisk):
                    R = max([rwake,rdisk])
                    r = min([rwake,rdisk])
                    x = (delta_span**2-r**2+R**2)/(2*np.abs(delta_span))
                    Aw = ( area_circle_reflex(r,x-np.abs(delta_span))
                          +area_circle_segment(R,x) )
                else:
                    Aw = np.pi*min([rwake,rdisk])**2

                induction = (1-np.sqrt(1-turbk.Ct))/2.
                Iadded = (0.73 * induction**(0.8325)
                               * abl.TI**(0.0325)
                               * (delta_str/turbk.D)**(-0.32) )
                TIadded[k] = Aw/(np.pi*rdisk**2)*Iadded
        TI[i] = np.sqrt(abl.TI**2 + (np.max(TIadded))**2)

    #Unsort turbines
    inv_order = np.argsort(order)
    return TI[inv_order]

def gauss_A(turbines,abl,TI):
    Nt = len(turbines)                  #Number of turbines
    Uinf = abl.U1
    Vinf = abl.V1
    Nq = 16                             #Number of quadrature pints
    e_str = e_streamwise(Uinf,Vinf)     #Unit vector along the wind direction
    e_span = e_spanwise(Uinf,Vinf)      #Unit vector in cross wind direction
    A = np.zeros((Nt,Nt))
    for i, turbi in enumerate(turbines):
        A[i,i] = 1.0
        #Loop over all other turbines to collect wake effects
        for k, turbk in enumerate(turbines):
            if k==i: continue           #Turbine i does not affect itself
            kwake = abl.kwake(TI[k])
            #Loop over quadrature points
            for n in range(Nq):
                wn  = 1.0/Nq
                phi = 2*np.pi*n/Nq
                r   = np.sqrt((3.+(1.-2*(n%2))*np.sqrt(3.))/6.)*turbi.D/2.0
                xn  = turbi.x + r*np.cos(phi)*e_span[0]
                yn  = turbi.y + r*np.cos(phi)*e_span[1]
                zn  = r*np.sin(phi) #Assuming all turbines have same zh
                #Vector from turbine K to point n on turbine I
                KI = np.array([xn-turbk.x,yn-turbk.y])
                #Streamwise distance between turbine K and turbine I along e_str
                #Positive if I is downstream of K 
                #Negative if I is upstream of K
                delta_str = np.dot(KI,e_str)
                #Spanwise distance between turbine K and turbine I along e_span
                #Has a sign but wake model is axisymmetric
                delta_span = np.dot(KI,e_span)
                #Compute wakedeficit at turbine I due to turbine K
                #minD = minimum distance between turbines to have wake interaction
                #(if x<minD, the wake deficit is not defined)
                beta = 0.5*(1+np.sqrt(1-turbk.Ct))/np.sqrt(1-turbk.Ct)
                eps  = 0.2*np.sqrt(beta)
                minD = 1./kwake*(np.sqrt(turbk.Ct/8.)-eps)*turbk.D
                if delta_str>minD:
                    A[i,k] += wn*gauss_wakedeficit(delta_str,delta_span,zn,
                                                   turbk.D,turbk.Ct,kwake)
                #Ghost turbine?
    return A

def gauss_Ajac(turbines,abl,TI):
    Nt = len(turbines)                  #Number of turbines
    Uinf = abl.U1
    Vinf = abl.V1
    Nq = 16                             #Number of quadrature pints
    e_str  = e_streamwise(Uinf,Vinf)    #Unit vector along the wind direction
    e_span = e_spanwise(Uinf,Vinf)      #Unit vector in cross wind direction
    E_str  = e_str_jac(Uinf,Vinf)       #Jacobian of the streamwise unit vector
    E_span = e_span_jac(Uinf,Vinf)      #Jacobian of the spanwise unit vector
    dAdu = np.zeros((Nt,Nt))
    dAdv = np.zeros((Nt,Nt))
    for i, turbi in enumerate(turbines):
        #dAdu[i,i] = 0.0
        #dAdv[i,i] = 0.0
        #Loop over all other turbines to collect wake effects
        for k, turbk in enumerate(turbines):
            if k==i: continue           #Turbine i does not affect itself
            kwake = abl.kwake(TI[k])
            #Loop over quadrature points
            for n in range(Nq):
                wn  = 1.0/Nq
                phi = 2*np.pi*n/Nq
                r   = np.sqrt((3+(1-2*(n%2))*np.sqrt(3))/6)*turbi.D/2.0
                xn  = turbi.x + r*np.cos(phi)*e_span[0]
                yn  = turbi.y + r*np.cos(phi)*e_span[1]
                zn  = r*np.sin(phi) #Assuming all turbines have same zh
                #Vector from turbine K to point n on turbine I
                KI = np.array([xn-turbk.x,yn-turbk.y])
                #Streamwise distance between turbine K and turbine I along e_str
                #Positive if I is downstream of K 
                #Negative if I is upstream of K
                delta_str = np.dot(KI,e_str)
                #Spanwise distance between turbine K and turbine I along e_span
                #Has a sign but wake model is axisymmetric
                delta_span = np.dot(KI,e_span)
                #Differentiate delta_str and delta_span
                dstr_jac = np.dot(KI.T,E_str)+r*np.cos(phi)*np.dot(E_span.T,e_str)
                dspan_jac= np.dot(KI.T,E_span)+r*np.cos(phi)*np.dot(E_span.T,e_span)
                #Derivative of wakedeficit at turbine I due to turbine K
                #minD = minimum distance between turbines to have wake interaction
                #(if x<minD, the wake deficit is not defined)
                beta = 0.5*(1+np.sqrt(1-turbk.Ct))/np.sqrt(1-turbk.Ct)
                eps  = 0.2*np.sqrt(beta)
                minD = 1./kwake*(np.sqrt(turbk.Ct/8.)-eps)*turbk.D
                if delta_str>minD:
                    defjac = gauss_wakedeficit_jac(delta_str,delta_span,zn,
                                                   turbk.D,turbk.Ct,kwake)
                    dAdu[i,k] += wn*(defjac[0]*dstr_jac[0]+defjac[1]*dspan_jac[0])
                    dAdv[i,k] += wn*(defjac[0]*dstr_jac[1]+defjac[1]*dspan_jac[1])
                #Ghost turbine?
    return dAdu,dAdv

def gauss_wakedeficit(x,y,z,d0,Ct,kwake):
    beta = 0.5*(1+np.sqrt(1-Ct))/np.sqrt(1-Ct)
    eps = 0.2*np.sqrt(beta)
    sigma_d0 = kwake*x/d0+eps
    deficit = ( (1-np.sqrt(1-Ct/(8*sigma_d0**2)))
                *np.exp(-1./(2*sigma_d0**2)*( (z/d0)**2+(y/d0)**2) ) )
    return deficit

def gauss_wakedeficit_jac(x,y,z,d0,Ct,kwake):
    beta = 0.5*(1+np.sqrt(1-Ct))/np.sqrt(1-Ct)
    eps = 0.2*np.sqrt(beta)
    sigma_d0 = kwake*x/d0+eps
    defdx  = (-Ct*kwake/(8*d0*sigma_d0**3)/np.sqrt(1-Ct/(8*sigma_d0**2))*
             np.exp(-1./(2*sigma_d0**2)*( (z/d0)**2+(y/d0)**2) ) )
    defdx += (gauss_wakedeficit(x,y,z,d0,Ct,kwake)*
                kwake/(d0*sigma_d0**3)*( (z/d0)**2+(y/d0)**2) )
    defdy  = -gauss_wakedeficit(x,y,z,d0,Ct,kwake)*y/(d0**2*sigma_d0**2)
    return np.array([defdx,defdy])

def area_circle_segment(R,d):
    return R**2*np.arccos(d/R)-d*np.sqrt(R**2-d**2)
def area_circle_reflex(R,d):
    alpha = np.arccos(d/R)
    return (np.pi-alpha)*R**2+d*np.sqrt(R**2-d**2)

def jensen(turbines,Uinf,Vinf,kwake):
#Depreciated
    Sinf = np.sqrt(Uinf**2+Vinf**2)   #Wind speed
    e_str = np.array([Uinf,Vinf])/Sinf #Unit vector along the wind direction
    e_span = np.array([-Vinf,Uinf])/Sinf #Unit vector in cross wind direction
    S = np.zeros((len(turbines)))
    Ftu = np.zeros((len(turbines)))
    Ftv = np.zeros((len(turbines)))
    for i, turbi in enumerate(turbines):
        #Compute superposition of wake deficits
        wakedeficit = 0.
        for turbk in turbines:
            #Vector from turbine K to turbine I
            KI = np.array([turbi.x-turbk.x,turbi.y-turbk.y])
            #Streamwise distance between turbine K and turbine I along e_str
            #Positive if I is downstream of K 
            #Negative if I is upstream of K
            delta_str = np.dot(KI,e_str)
            #Spanwise distance between turbine K and turbine I along e_span
            #Has a sign but wake model is axisymmetric
            delta_span = np.dot(KI,e_span)
            #Compute wakedeficit at turbine I due to turbine K
            #Step function so that only upstream turbines K influence turbine I
            #Pulse function to account for the width of the wake (check whether
            #the spanwise separation between I and K is smaller than the half-width
            #of the wake)
            wakedeficit += ( (1.0-np.sqrt(1.0-turbk.Ct))
                              /(1+2*kwake*delta_str/turbk.D)**2
                              * step(delta_str)
                              * pulse(delta_span,turbk.D/2.0+kwake*delta_str) )**2
        #Velocity and Force at turbine i
        S[i] = Sinf*(1-np.sqrt(wakedeficit))
        Ftu[i] = 0.5* turbi.Ct * turbi.rotorarea * S[i]**2 * e_str[0]
        Ftv[i] = 0.5* turbi.Ct * turbi.rotorarea * S[i]**2 * e_str[1]
    return Ftu,Ftv,S

def nowake(turbines,abl):
    Uinf = abl.U1
    Vinf = abl.V1
    Sinf = abl.S1
    Ft = np.zeros((len(turbines),2))
    for index,turb in enumerate(turbines):
        Ft[index,0] = 0.5* turb.Ct * turb.rotorarea * Sinf*Uinf
        Ft[index,1] = 0.5* turb.Ct * turb.rotorarea * Sinf*Vinf
    return Ft

def nowake_jac(turbines,abl):
    Uinf = abl.U1
    Vinf = abl.V1
    Sinf = abl.S1
    Ftjac = np.zeros((len(turbines),2,2))
    for index,turb in enumerate(turbines):
        Ftjac[index,0,0] = 0.5* turb.Ct * turb.rotorarea * (Sinf+Uinf**2/Sinf)
        Ftjac[index,0,1] = 0.5* turb.Ct * turb.rotorarea * (Uinf*Vinf/Sinf)
        Ftjac[index,1,0] = 0.5* turb.Ct * turb.rotorarea * (Uinf*Vinf/Sinf)
        Ftjac[index,1,1] = 0.5* turb.Ct * turb.rotorarea * (Sinf+Vinf**2/Sinf)
    return Ftjac

def e_streamwise(Uinf,Vinf):
    '''Unit vector along the wind direction'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    return np.array([Uinf,Vinf])/Sinf
def e_spanwise(Uinf,Vinf):
    '''Unit vector in cross wind direction'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    return np.array([-Vinf,Uinf])/Sinf
def e_str_jac(Uinf,Vinf):
    '''Jacobian of the streamwise unit vector'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    J = 1./Sinf**3 * np.ones((2,2))
    J[0,0] *= Vinf**2
    J[0,1] *= -Uinf*Vinf
    J[1,0] *= -Uinf*Vinf
    J[1,1] *= Uinf**2
    return J
def e_span_jac(Uinf,Vinf):
    '''Jacobian of the spanwise unit vector'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    J = 1./Sinf**3 * np.ones((2,2))
    J[0,0] *= Uinf*Vinf
    J[0,1] *= -Uinf**2
    J[1,0] *= Vinf**2
    J[1,1] *= -Uinf*Vinf
    return J
