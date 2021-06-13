#!/usr/bin/env python

'''
Run TLM and gather statistics about memory usage
'''
__author__ = "Lanzilao Luca"
__date__ = "January 17, 2020"

import numpy as np
import time
from tlmpy import TLM
from tlmpy import TLMForcing
from tlmpy import TLM_tools
import tracemalloc

def TLMrun(): 
    #------------------------------------------------------#
    #-------------- Step 1: define ABL --------------------#
    #------------------------------------------------------#

    #Input parameters
    fc    = 1.0e-4                                      #Coriolis parameter [1/s]
    th0   = 288.15                                      #Reference temperature [K]
    dthdz = 1.0e-3                                      #free atmosphere lapse rate [K/m]
    g     = 9.81                                        #Gravitational acceleration [m/s^2]
    N     = np.sqrt(g/th0*dthdz)                        #Brunt Vaisala frequency [1/s]
    kappa = 0.41                                        #Von Karman constant [-]
    utau  = 0.666                                       #Friction velocity [m/s]
    TI    = 0.12                                        #Ambient turbulent intensity [-]
    H1    = 240.                                        #Wind-farm layer height (2*zh) [m]    
    #reference subcritical flow case - Fr = 0.9
    dth   = 6.824                                       #Inversion strength [K]
    #reference supercritical flow case - Fr = 1.1
    #dth = 4.510                                        #Inversion strength [K]    
    hstar = 0.15
    z0_h  = 1.e-4    
    Cg    = TLM_tools.Cg_cubic(hstar,z0_h,kappa)        # Geostrophic drag Cg = utau/G
    alpha = TLM_tools.alpha_cubic(hstar,z0_h,kappa)     # Geostrophic wind angle
    h     = hstar * utau / fc                           # Friction velocity [m/s]
    G     = utau/Cg                                     # Geostrophic wind speed [m/s]
    FA_condition = 'non-hydrostatic'                    #Free atmosphere condition (hydro,non-hydro) 
    
    #Generate ABL object using "analytic cubic" method,
    #which uses the analytical boundary-layer model of
    #Nieuwstadt (1883) with a cubic eddy-viscosity profile
    abl = TLM.ABL(input='analytic_cubic',
                    dth=dth,
                    fc=fc,
                    N=N,
                    G=G,
                    alpha=alpha,
                    kappa=kappa,
                    utau=utau,
                    h=h,
                    H1=H1,
                    TI=TI,
                    FA_condition=FA_condition
                    )
    
    #Rotate abl so that wind is aligned in the wind-farm layer
    abl.rotate(abl.WD1*np.pi/180.0)
    
    #------------------------------------------------------#
    #--------- Step 2: define numerical grid --------------#
    #------------------------------------------------------#
    
    #Numerical parameters
    Nx = 2000           #Grid points in x-direction
    Lx = 1.e6           #Grid size in x-direction [m]
    Ny = 800            #Grid points in y-direction
    Ly = 0.4e6          #Grid size in y-direction [m]
    
    #Generate 2D grid object
    grid = TLM.Stat2Dgrid(Lx,Nx,Ly,Ny)
    
    
    #------------------------------------------------------#
    #------------- Step 3: define wind farm ---------------#
    #------------------------------------------------------#
    
    #Input parameters
    D = 154.            #Turbine diameter [m]
    Ntx = 18            #Number of turbine rows
    Nty = 27            #Number of turbine columns
    Ct  = 0.8           #Turbine thrust coefficient
    zh  = 120.          #Turbine hub height [m]
    Lfilter = 1000.     #Gaussian filter length
    
    
    #original
    Lwfx = Ntx/9.0e-4                               #Wind-farm length                        
    Lwfy = Nty/9.0e-4                               #Wind-farm width
    xs = np.linspace(0,Lwfx,Ntx,endpoint=False)
    ys,dy = np.linspace(0,Lwfy,Nty,endpoint=False,retstep=True)
    X,Y = np.meshgrid(xs,ys,indexing='ij')
    Y[1::2,:] += dy/2.                              #Staggered grid
    
    #Generate wind farm object
    #Use Gaussian wake model of Niayifar (2015)
    #Use velocity at 10D upstream as inflow velocity
    Cts = Ct*np.ones((X.size))
    Ds  = D*np.ones((X.size))
    zhs = zh*np.ones((X.size)) 
    forcing = TLMForcing.WF(grid,np.ravel(X)+Lx/2.,
                            np.ravel(Y)+Ly/2.,
                            Ds,Cts,zhs,Lfilter=Lfilter,
                            wakemodel='gauss',
                            coupling_location='upstream_10',
                            )
    
    #------------------------------------------------------#
    #- Step 4: create TLM model from components and solve -#
    #------------------------------------------------------#
    
    #Create static 2D model
    model = TLM.S2Dmodel(grid,forcing,abl,purefriction=False)
    
    #Solve linear system
    result = model.solve(method='gcrotmk',verbose=True,WFfeedback=True,convergence_info=False)
       
    #Evaluate wind-farm power output [GW] 
    P = forcing.Ptot(abl,grid,result['u1r'],result['v1r'])/1.0e9
        
    return P

print('Start script')
start = time.time()  

n=1
for k in range(n):
    tracemalloc.start()
    TLMrun()
    current, peak = tracemalloc.get_traced_memory()
    print('Current memory usage is ',current/10**6,' MB') 
    print('Peak memory usage was ',peak/10**6,'MB \n')
    
    memory_tm = tracemalloc.get_tracemalloc_memory()
    print('Memory usage of tracemalloc is ',memory_tm/10**6,' MB')
    tracemalloc.stop()
    
end = time.time()
print('Time to run the script was',(end-start)/n,'s')




        
        
        
        
        
        
    