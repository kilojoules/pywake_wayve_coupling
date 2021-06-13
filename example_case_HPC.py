#!/usr/bin/env python
'''
FOR SuperComputer run: -Set parallel=False in TLM_tools.evaluate_M
                       -Set M = np.linalg.solve(P,X)
'''
__author__ = "Luca Lanzilao"
__date__ = "February 23, 2020"

import numpy as np
import time
from tlmpy import TLM
from tlmpy import TLMForcing
from tlmpy import TLM_tools
from mpi4py import MPI

comm = MPI.COMM_WORLD
np_pp = comm.Get_size()
rank   = comm.Get_rank()

def TLMrun(parameter_loc): 
    #------------------------------------------------------#
    #-------------- Step 1: define ABL --------------------#
    #------------------------------------------------------#

    #Input parameters
    fc  = 1.0e-4                  #Coriolis parameter [1/s]
    th0   = 288.15                #Reference temperature [K]
    dthdz   = 1.0e-3              #Free atmosphere lapse rate [K/m]
    g     = 9.81                  #Gravitational acceleration [m/s^2]
    N = np.sqrt(g/th0*dthdz)      #Brunt Vaisala frequency [1/s]
    #G   = 15.4                   #Geostrophic wind speed [m/s]
    #alpha = -0.3635              #Geostrophiv wind angle [rad]
    kappa = 0.41                  #Von Karman constant [-]
    utau  = 0.666                 #Friction velocity [m/s]
    #h     = 1000.                #Boundary-layer height [m]
    H1    = 240.                  #Wind-farm layer height [m]
    #H2    = h - H1               #Upper-layer height [m]
    TI    = 0.12                  #Ambient turbulent intensity [-]
    
    #reference subcritical flow case - Fr = 0.9
    dth    = parameter_loc#6.824         #Inversion strength [K]
    #reference supercritical flow case - Fr = 1.1
    #dth = 4.510                         #Inversion strength [K]
    g      = 9.81                        #gravity [m/s^2]    
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
    #abl.rotate((45*abl.WD1*np.pi)/180.0)
    
    #Velocity scale
    #Ub = 1/np.sqrt(H1/(h*abl.U1**2) + H2/(h*abl.U2**2))       
    #Froude number - characteristic for interfacial gravity waves
    #Fr = Ub/np.sqrt(gprime*h)
    #PN number - characteristic for internal gravity waves
    #PN = Ub**2/(N*h*abl.S3)
    
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
    
    #Create lay-out
    #sx = 7.215             #turbine x-spacing adimensionalised with D
    #sy = 7.215             #turbine y-spacing adimensionalised with D
    #Sx = sx*D
    #Sy = sy*D
    #Lwfx = Ntx*Sx                                                    #windfarm width
    #Lwfy = Nty*Sy 
    #original
    Lwfx = Ntx/9.0e-4
    Lwfy = Nty/9.0e-4
    xs = np.linspace(0,Lwfx,Ntx,endpoint=False)
    ys,dy = np.linspace(0,Lwfy,Nty,endpoint=False,retstep=True)
    X,Y = np.meshgrid(xs,ys,indexing='ij')
    Y[1::2,:] += dy/2.    #Staggered grid
    
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
    
    #Solve using LGMRES
    result = model.solve(method='gcrotmk',verbose=True,WFfeedback=True,convergence_info=False)
        
    P = forcing.Ptot(abl,grid,result['u1r'],result['v1r'])/1.0e9
        
    return P,parameter_loc

start = time.time()

dth = np.linspace(1,36,36)#np.array([6.824])#

# distribute over the different procs
if rank == 0:
    chunks = [[] for _ in range(np_pp)]      #List of empty array like [[],[],[],...,[]]
    for i, chunk in enumerate(dth):
        chunks[i % np_pp].append(chunk)
else:
    chunks = None

parameters_loc = comm.scatter(chunks, root=0)    #I send all single parameter to different processors


for parameter_loc in parameters_loc:
    P,dth  = TLMrun(parameter_loc)


output = {'Power':P,'dth':dth}

output = comm.gather(output,root=0) #Gather data from all processors. It stores all in an array

#print only in rank0
if rank==0:   
    for k in range(np_pp):
        print('With strength',output[k]['dth'],'K the power is ', output[k]['Power'])
    
    end = time.time()
    print('Time to run the script was',end-start,'s')
    
    
    