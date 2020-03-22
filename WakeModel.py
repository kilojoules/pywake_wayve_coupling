#!/usr/bin/env python

'''
Wake models
'''

import numpy as np
from py4sp.mypy import step
from py4sp.mypy import pulse
from numba import njit
import numba

def gauss(turbines,abl):
    '''
    Gaussian wake model

    Parameters
    ----------
    turbines: list of turbine objects (defined in TLMForcing.py)
        list with the wind turbines
    abl: ABL object (defined in TLMForcing.py)
        atmospheric state

    Returns
    -------
    S: 1d numpy array
        inflow velocities at the turbines
    
    _: 2d numpy array with shape (Nturb,2)
        Jacobian of inflow velocities
    '''
    Nt = len(turbines)
    Uinf = abl.U1
    Vinf = abl.V1
    Sinf = abl.S1
    TIinf = abl.TI
    Nq = 16                                          #Number of quadrature pints
    e_str = e_streamwise(Uinf,Vinf)                  #Unit vector along the wind direction
    e_span = e_spanwise(Uinf,Vinf)                   #Unit vector in cross wind direction
    E_str  = e_str_jac(Uinf,Vinf)                    #Jacobian of the streamwise unit vector
    E_span = e_span_jac(Uinf,Vinf)                   #Jacobian of the spanwise unit vector
    x = np.array([turbines[k].x for k in range(Nt)])
    y = np.array([turbines[k].y for k in range(Nt)])
    D = np.array([turbines[k].D for k in range(Nt)])
    Ct = np.array([turbines[k].Ct for k in range(Nt)])
    zh = np.array([turbines[k].zhs for k in range(Nt)])
    
    #for turbulence intensity
    #Sort turbines along wind direction - for TI
    coordinates = np.concatenate([x,y]).reshape(Nt,2,order='F')
    dist = np.dot(coordinates,e_str)
    order = np.argsort(dist)
    x_sort = np.array([turbines[i].x for i in order])
    y_sort = np.array([turbines[i].y for i in order])
    D_sort = np.array([turbines[i].D for i in order])
    Ct_sort = np.array([turbines[i].Ct for i in order])
    
    #Compute TI at turbine location
    TI = evaluate_TI(Nt,e_str,e_span,order,x_sort,y_sort,D_sort,Ct_sort,TIinf)  
    #Compute turbine interaction matrix and its derivatives
    A,dAdu,dAdv = evaluate_A_Ajac(Nt,Nq,e_str,e_span,E_str,E_span,x,y,Ct,D,zh,TI)
    B  = Sinf*np.ones(Nt)                              #Vector with undisturbed turbine inflow velocities
    S  = np.linalg.solve(A,B)                          #Vector with turbine inflow velocities - Solve linear system
     
    dSinfdu = Uinf/Sinf*np.ones((Nt))
    dSinfdv = Vinf/Sinf*np.ones((Nt)) 
    dSdu = np.linalg.solve(A,dSinfdu-np.dot(dAdu,S))   #Vector with u-derivative of turbine inflow velocities - Solve linear system    
    dSdv = np.linalg.solve(A,dSinfdv-np.dot(dAdv,S))   #Vector with v-derivative of turbine inflow velocities - Solve linear system   

    return S,np.vstack([dSdu,dSdv]).T


@njit(parallel=False)
def evaluate_TI(Nt,e_str,e_span,order,x,y,D,Ct,TIinf): 
    '''
    Turbulence intensity model of Niayifar and Porte-Agel (2016) in Numba syntax

    Turbulence intensity is assumed not to depend on the perturbation velocities so
    that, contrary to the Gaussian wake model itself, the turbulence intensity model
    can be calculated by ordering the turbines and then sweeping through the farm

    Parameters
    ----------
    Nt: float
        Total wind turbine number
    e_str: 1d numpy array
        Unit vector along the wind direction
    e_span: 1d numpy array
        Unit vector in cross wind direction
    order: 1d numpy array
        Wind turbine index ordered according to wind direction
    x: 1d numpy array
        Wind turbine x-coordinate
    y: 1d numpy array
        Wind turbine y-coordinate
    D: 1d numpy array
        Wind turbine diameter
    Ct: 1d numpy array
        Wind turbine thrust coefficient
    TIinf: float
        Free stream turbulence intensity

    Returns
    -------
    TI: 1d numpy array
        Local turbulent intensity at the turbines
    '''
    TI = np.zeros((Nt))
    TI[0] = TIinf
    for i in numba.prange(1,Nt):
        TIadded = np.zeros((i))
        maxTIadded = 0
        #Compute added streamwise turbulent intensity induced by turbine k
        #at turbine i        
        for k in numba.prange(0,i):
            kwake = 0.3837*TI[k]+0.003678
            #Vector from turbine K to point n on turbine I
            KI = np.array([x[i]-x[k],y[i]-y[k]])
            #Streamwise distance between turbine K and turbine I along e_str
            #Positive if I is downstream of K 
            #Negative if I is upstream of K
            delta_str = np.dot(KI,e_str)
            #Spanwise distance between turbine K and turbine I along e_span
            #Has a sign but wake model is axisymmetric
            delta_span = np.dot(KI,e_span)
            #Turbine k lies upstream of Turbine i
            if delta_str>0:# and delta_str<15*turbk.D:
                beta = 0.5*(1+np.sqrt(1-Ct[k]))/np.sqrt(1-Ct[k])
                eps = 0.2*np.sqrt(beta)
                sigma = kwake*delta_str+D[k]*eps
                rwake = 2*sigma
                rdisk = D[i]/2.0
                #TI wake of k intersects with rotor i
                #the wake start at the edge of the turbine rotor, so the width of the wake
                #is given by rdisk + the width of the wake, so rwake
                #If the lateral distance between turbine is greater than rwake+rdisk,
                #then there is no overlap
                maximum = 0
                minimum = 0
                if rwake>=rdisk:
                    maximum = rwake
                    minimum = rdisk
                else:
                    maximum = rdisk
                    minimum = rwake
                if np.abs(delta_span)>=(rwake+rdisk):
                    Aw = 0. #No overlap
                elif np.abs(delta_span)>=maximum:
                    xt = (delta_span**2-rdisk**2+rwake**2)/(2*np.abs(delta_span))
                    Aw = ( area_circle_segment(rwake,xt)
                          +area_circle_segment(rdisk,np.abs(delta_span)-xt) )
                elif np.abs(delta_span)>=np.abs(rwake-rdisk):
                    R = maximum
                    r = minimum 
                    xt = (delta_span**2-r**2+R**2)/(2*np.abs(delta_span))
                    Aw = ( area_circle_reflex(r,xt-np.abs(delta_span))
                          +area_circle_segment(R,xt) )
                else:
                    Aw = np.pi*minimum**2  #Area of overlapping - The overlap cannot be smaller than rdisk

                induction = (1-np.sqrt(1-Ct[k]))/2.
                Iadded = (0.73 * induction**(0.8325)
                               * TIinf**(0.0325)
                               * (delta_str/D[k])**(-0.32) )
                TIadded[k] = Aw/(np.pi*rdisk**2)*Iadded
                if TIadded[k] > maxTIadded:
                    maxTIadded = TIadded[k]
        TI[i] = np.sqrt(TIinf**2 + (maxTIadded)**2)

    #Unsort turbines
    inv_order = np.argsort(order)
    return TI[inv_order]

@njit(parallel=False)
def evaluate_A_Ajac(Nt,Nq,e_str,e_span,E_str,E_span,x,y,Ct,D,zh,TI):
    '''
    Compute system matrix A of Gaussian wake model
    Compute Jacobian of system matrix A of Gaussian wake model
    
    Parameters
    ----------
    Nt: float
        Total wind turbine number
    Nq: float
        Number of quadrature point on wind turbine rotor disk
    e_str: 1d numpy array
        Unit vector along the wind direction
    e_span: 1d numpy array
        Unit vector in cross wind direction
    E_str: 1d numpy array
        Jacobian of the streamwise unit vector
    E_span: 1d numpy array
        Jacobian of the spanwise unit vector
    x: 1d numpy array
        Wind turbine x-coordinate
    y: 1d numpy array
        Wind turbine y-coordinate
    D: 1d numpy array
        Wind turbine diameter
    Ct: 1d numpy array
        Wind turbine thrust coefficient
    zh: 1d numpy array
        Wind turbine hub height
    TI: 1d numpy array
        Turbulence intensity at turbine location

    Returns
    -------
    A: numpy array with shape (Nt,Nt)
        system matrix of Gaussian wake model
    dAdu,dAdv: numpy array with shape (Nt,Nt)
        Jacobian of system matrix of Gaussian wake model w.r.t. u and v
    '''
    A = np.identity((Nt))
    dAdu = np.zeros((Nt,Nt))
    dAdv = np.zeros((Nt,Nt))
    for i in numba.prange(Nt):
        #Loop over all other turbines to collect wake effects
        for k in numba.prange(Nt):
            kwake = 0.3837*TI[k]+0.003678
            #minD = minimum distance between turbines to have wake interaction
            #(if x<minD, the wake deficit is not defined)
            beta = 0.5*(1+np.sqrt(1-Ct[k]))/np.sqrt(1-Ct[k])
            eps  = 0.2*np.sqrt(beta)
            minD = 1./kwake*(np.sqrt(Ct[k]/8.)-eps)*D[k]
            #Vector from turbine K to center of turbine I
            KI = np.array([x[i]-x[k],y[i]-y[k]])
            #Streamwise distance between turbine K and turbine I along e_str
            #Positive if I is downstream of K 
            #Negative if I is upstream of K
            delta_str = np.dot(KI,e_str)
            #test distance between center of turbine I and K.
            #If yaw angle of both turbines is equal, delta_str is identical for all
            #quadrature points.
            if delta_str<=0 or delta_str<=minD: continue
            #vertical distance between turbine hub heigths 
            delta_z = np.abs(zh[k] - zh[i])
            #Loop over quadrature points
            for n in numba.prange(Nq):
                wn  = 1.0/Nq
                phi = 2*np.pi*n/Nq
                r   = np.sqrt((3.+(1.-2*(n%2))*np.sqrt(3.))/6.)*D[i]/2.0
                xn  = x[i] + r*np.cos(phi)*e_span[0]
                yn  = y[i] + r*np.cos(phi)*e_span[1]
                zn  = delta_z + r*np.sin(phi)*np.sign(-(zh[k] - zh[i])) #Assuming turbines with different zh
                if delta_z == 0:
                    zn  = r*np.sin(phi)                                 #Assuming all turbines have same zh
                #Vector from turbine K to point n on turbine I
                KI = np.array([xn-x[k],yn-y[k]])
                #Streamwise distance between turbine K and turbine I along e_str
                #Positive if I is downstream of K 
                #Negative if I is upstream of K
                delta_str = np.dot(KI,e_str)
                #Spanwise distance between turbine K and turbine I along e_span
                #Has a sign but wake model is axisymmetric
                delta_span = np.dot(KI,e_span)
                #Compute wakedeficit at turbine I due to turbine K
                A[i,k] = A[i,k] + wn*gauss_wakedeficit(delta_str,delta_span,zn,D[k],Ct[k],kwake,beta,eps)
                #Compute A_jac
                dstr_jac = np.dot(KI.T,E_str)+r*np.cos(phi)*np.dot(E_span.T,e_str)
                dspan_jac= np.dot(KI.T,E_span)+r*np.cos(phi)*np.dot(E_span.T,e_span)
                #Derivative of wakedeficit at turbine I due to turbine K
                defjac = gauss_wakedeficit_jac(delta_str,delta_span,zn,D[k],Ct[k],kwake,beta,eps)
                dAdu[i,k] = dAdu[i,k] + wn*(defjac[0]*dstr_jac[0]+defjac[1]*dspan_jac[0])
                dAdv[i,k] = dAdv[i,k] + wn*(defjac[0]*dstr_jac[1]+defjac[1]*dspan_jac[1])
    
    return A,dAdu,dAdv

@njit
def gauss_wakedeficit (x,y,z,d0,Ct,kwake,beta,eps):
    '''
    Wake deficit function - decorate with @njit, which speeds up the calculation

    Parameters
    ----------
    x,y,z: float
        relative streamwise, spanwise and vertical distance in the wake
    d0: float
        rotor diameter
    Ct: float
        thrust coefficient
    kwake: float
        wake expansion coefficient
    beta: float
        Parameter defined in Niayifar (2015)
    eps: float
        Parameter defined in Niayifar (2015)

    Returns
    -------
    deficit: float
        velocity deficit
    '''
    sigma_d0 = kwake*x/d0+eps
    deficit = ( (1-np.sqrt(1-Ct/(8*sigma_d0**2)))
                *np.exp(-1./(2*sigma_d0**2)*( (z/d0)**2+(y/d0)**2) ) )
    return deficit

@njit
def gauss_wakedeficit_jac(x,y,z,d0,Ct,kwake,beta,eps):
    '''
    Jacobian of wake deficit function - decorate with @njit, which speeds up the calculation

    Parameters
    ----------
    x,y,z: float
        relative streamwise, spanwise and vertical distance in the wake
    d0: float
        rotor diameter
    Ct: float
        thrust coefficient
    kwake: float
        wake expansion coefficient
    beta: float
        Parameter defined in Niayifar (2015)
    eps: float
        Parameter defined in Niayifar (2015)

    Returns
    -------
    _: numpy array with size (2,)
        Jacobian of velocity deficit w.r.t. x and y
    '''
    sigma_d0 = kwake*x/d0+eps
    defdx  = (-Ct*kwake/(8*d0*sigma_d0**3)/np.sqrt(1-Ct/(8*sigma_d0**2))*
             np.exp(-1./(2*sigma_d0**2)*( (z/d0)**2+(y/d0)**2) ) +
             (gauss_wakedeficit(x,y,z,d0,Ct,kwake,beta,eps)*
                kwake/(d0*sigma_d0**3)*( (z/d0)**2+(y/d0)**2) ) )
    defdy  = -gauss_wakedeficit(x,y,z,d0,Ct,kwake,beta,eps)*y/(d0**2*sigma_d0**2)
    return np.array([defdx,defdy])

@njit
def area_circle_segment(R,d):
    '''Area of a circle segment when the angle is less than 180°'''
    return R**2*np.arccos(d/R)-d*np.sqrt(R**2-d**2)

@njit
def area_circle_reflex(R,d):
    '''Area of a circle segment when the angle is greater than 180°'''
    alpha = np.arccos(d/R)
    return (np.pi-alpha)*R**2+d*np.sqrt(R**2-d**2)

def nowake(turbines,abl):
    '''
    No wake model

    No wake effects so inflow velocities are just the unperturbed wind speed

    Parameters
    ----------
    turbines: list of turbine objects (defined in TLMForcing.py)
        list with the wind turbines
    abl: ABL object (defined in TLMForcing.py)
        atmospheric state

    Returns
    -------
    S: 1d numpy array
        inflow velocities at the turbines
    _: float
        Coherent syntax with WF.preprocess
    '''
    Nt = len(turbines)
    return abl.S1*np.ones(Nt),np.zeros((Nt,2))

@njit
def nowake_jac(turbines,abl,Nt):
    '''
    Jacobian of the no wake model

    Parameters
    ----------
    turbines: list of turbine objects (defined in TLMForcing.py)
        list with the wind turbines
    abl: ABL object (defined in TLMForcing.py)
        atmospheric state

    Returns
    -------
    _: 2d numpy array with shape (Nturb,2)
        Jacobian of inflow velocities
    '''
    dSdu = abl.U1/abl.S1*np.ones((Nt))
    dSdv = abl.V1/abl.S1*np.ones((Nt))
    return np.vstack([dSdu,dSdv]).T

@njit
def e_streamwise(Uinf,Vinf):
    '''Unit vector along the wind direction'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    return np.array([Uinf,Vinf])/Sinf

@njit
def e_spanwise(Uinf,Vinf):
    '''Unit vector in cross wind direction'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    return np.array([-Vinf,Uinf])/Sinf

@njit
def e_str_jac(Uinf,Vinf):
    '''Jacobian of the streamwise unit vector'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    J = 1./Sinf**3 * np.ones((2,2))
    J[0,0] *= Vinf**2
    J[0,1] *= -Uinf*Vinf
    J[1,0] *= -Uinf*Vinf
    J[1,1] *= Uinf**2
    return J

@njit
def e_span_jac(Uinf,Vinf):
    '''Jacobian of the spanwise unit vector'''
    Sinf = np.sqrt(Uinf**2+Vinf**2)
    J = 1./Sinf**3 * np.ones((2,2))
    J[0,0] *= Uinf*Vinf
    J[0,1] *= -Uinf**2
    J[1,0] *= Vinf**2
    J[1,1] *= -Uinf*Vinf
    return J

@njit(parallel=False)
def footprint(xs,ys,dx,dy,L,xloc,yloc):
    '''
    Wind turbine footprint in Numba syntax. Limit denotes the size of the 
    domain which contains the wind turbine footprint. Only the row,column 
    and value of the footprint are stored, to limit memory usage and to speed
    up the foorprint evaluation.
    Possible improvements:
        -express limit as function of the grid resolution

    Parameters
    ----------
    xs,ys: 1d numpy array
        x and y points on numerical domain
    dx,dy: 1d numpy array
        grid cell size
    xloc,yloc: scalar
        x and y location of the individual turbine

    Returns
    -------
    row: 1d numpy array
        row index of wind turbine footprint
    col: 1d numpy array
        column index of wind turbine footprint
    val: 1d numpy array
        value of wind turbine footprint
    '''
    #The value of limit can be reduced to 3 for grid resolution of 5 km and 
    #it may be increased if grid resolution higher than 250 meter are used
    limit=13
    R = np.zeros((limit*2,limit*2))
    val = np.zeros(((limit*2)**2))
    row = np.zeros(((limit*2)**2),dtype=np.int64)
    col = np.zeros(((limit*2)**2),dtype=np.int64)
    summ = 0
    count = 0    
    plus_i = int(xloc/dx-limit)
    plus_j = int(yloc/dy-limit)
    for i in numba.prange(limit*2):
        for j in numba.prange(limit*2):
            R[i,j] = 1./(np.pi*L**2)*np.exp(-((xs[plus_i+i]-xloc)**2+(ys[plus_j+j]-yloc)**2)/L**2)
            summ = summ + R[i,j]
    for i in numba.prange(limit*2):     
        for j in numba.prange(limit*2):
            val[count] = R[i,j]/(summ*dx*dy)
            if val[count] < 1.e-20:
                val[count]=0.
                count += 1
            else:
                row[count] = plus_i + i
                col[count] = plus_j + j
                count += 1
    return row[np.nonzero(row)],col[np.nonzero(col)],val[np.nonzero(val)]

def gauss_alternative(turbines,abl):
    '''
    Alternative implementation of Gaussian wake model (depreciated)

    Sort wind turbines and then advance through farm. Is faster than building
    a general matrix, but issues arise with the Jacobian when the order of turbines
    changes due to the perturbation
    '''
    TI = gauss_TI(turbines,abl)           #Compute TI at turbine locations
    Uinf = abl.U1
    Vinf = abl.V1
    Nt = len(turbines)                    #Number of turbines
    Nq = 16                               #Number of quadrature points
    Sinf = np.sqrt(Uinf**2+Vinf**2)       #Wind speed
    e_str = np.array([Uinf,Vinf])/Sinf    #Unit vector along the wind direction
    e_span = np.array([-Vinf,Uinf])/Sinf  #Unit vector in cross wind direction
    #Sort turbines along wind direction
    xs = np.array([turbines[i].x for i in range(Nt)])
    ys = np.array([turbines[i].y for i in range(Nt)])
    coordinates = np.concatenate([xs,ys]).reshape(Nt,2,order='F')
    dist = np.dot(coordinates,e_str)
    order = np.argsort(dist)
    turbines_sort = [turbines[i] for i in order]
    kwake = [abl.kwake(TI[i]) for i in order]
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
                    wake_deficit += S[k]*wn*gauss_wakedeficit(delta_str,delta_span,zn,turbk.D,turbk.Ct,kwake[k])
                    #wake_deficit += wn*gauss_wakedeficit(delta_str,delta_span,zn,turbk.D,turbk.Ct,kwake[k])
               
                #Ghost turbine?
        S[i] = Sinf - wake_deficit
        #S[i] = Sinf * (1 - wake_deficit)
    #Unsort turbines
    inv_order = np.argsort(order)
    return S[inv_order]

def gauss_TI(turbines,abl):
    '''
    Turbulence intensity model of Niayifar and Porte-Agel (2016)

    NEEDED ONLY IF gauss_alternative IS USED!

    Parameters
    ----------
    turbines: list of turbine objects (defined in TLMForcing.py)
        list with the wind turbines
    abl: ABL object (defined in TLMForcing.py)
        atmospheric state

    Returns
    -------
    TI: 1d numpy array
        local turbulent intensity at the turbines
    '''
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


def jensen(turbines,Uinf,Vinf,kwake):
    '''Jensen wake model (depreciated)'''
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

