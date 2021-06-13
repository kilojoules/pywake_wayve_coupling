#!/usr/bin/env python
'''
Solve depth-average RANS equations subject to wind-farm forcing
'''
__author__ = "Lanzilao Luca, Dries Allaerts"
__date__ = "January 17, 2020"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tlmpy import TLM
from tlmpy import TLMForcing
from tlmpy import TLM_tools
from py4sp import mypy

plot=False       #plot results

#======================================================================================================================#
'''
Define ABL
'''

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
Cg    = TLM_tools.Cg_cubic(hstar,z0_h,kappa)        #Geostrophic drag Cg = utau/G
alpha = TLM_tools.alpha_cubic(hstar,z0_h,kappa)     #Geostrophic wind angle
h     = hstar * utau / fc                           #Friction velocity [m/s]
G     = utau/Cg                                     #Geostrophic wind speed [m/s]
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

#======================================================================================================================#
'''
Define numerical grid
'''

#Numerical parameters
Nx = 2000           #Grid points in x-direction
Lx = 1.e6           #Grid size in x-direction [m]
Ny = 800            #Grid points in y-direction
Ly = 0.4e6          #Grid size in y-direction [m]

#Generate 2D grid object
grid = TLM.Stat2Dgrid(Lx,Nx,Ly,Ny)


#======================================================================================================================#
'''
Define windfarm
'''

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

#======================================================================================================================#
'''
create TLM model from components and solve
'''

#Create static 2D model
model = TLM.S2Dmodel(grid,forcing,abl,purefriction=False)

#Solve linear system
result = model.solve(method='gcrotmk',verbose=True,WFfeedback=True,convergence_info=False)
   
#Evaluate wind-farm power output [GW] 
P = forcing.Ptot(abl,grid,result['u1r'],result['v1r'])/1.0e9

#======================================================================================================================#
'''
Plot results
'''

if plot:    
    #------------------------------------------------------#
    #------------ Some example post processing ------------#
    #------------------------------------------------------#
    
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    
    #Plotting function that will be used later on
    def plot_field(name,field,grid,forcing,clabel,vmin=None,vmax=None,ax=None,f=None):
        norm = mypy.MidpointNormalize(midpoint=0)
        cmap = mpl.cm.get_cmap('RdBu_r')
        im = ax.pcolormesh((grid.xs-forcing.xcentre)/1.0e3,
                       (grid.ys-forcing.ycentre)/1.0e3,
                       np.transpose(field),
                       shading='gouraud',
                       norm=norm,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       rasterized=True)
        x1 = (forcing.xstart-forcing.xcentre)/1.0e3
        x2 = (forcing.xend-forcing.xcentre)/1.0e3
        y1 = (forcing.ystart-forcing.ycentre)/1.0e3
        y2 = (forcing.yend-forcing.ycentre)/1.0e3
        ax.plot([x1,x1],[y1,y2],'-k')
        ax.plot([x1,x2],[y2,y2],'-k')
        ax.plot([x2,x2],[y1,y2],'-k')
        ax.plot([x1,x2],[y1,y1],'-k')
        ax.set_xlim([-100,100])
        ax.set_ylim([-100,100])
        cbar = f.colorbar(im,ax=ax,shrink=1.0)
        cbar.set_label(clabel)
        cbar.set_clim(-np.max(np.abs(np.array([vmin,vmax]))),
                       np.max(np.abs(np.array([vmin,vmax]))))
        ax.set_xlabel(r'$x\;[\mathrm{km}]$')
        ax.set_ylabel(r'$y\;[\mathrm{km}]$')
        return
    
    #Difference in velocity magnitude in wind-farm layer
    M1 = -(abl.S1-np.sqrt((abl.U1+result['u1r'])**2 +
                          (abl.V1+result['v1r'])**2)) / abl.S1*100
    
    #Plot result
    f, axarr = plt.subplots(1,3,figsize=(19.2,4.8))
    f.subplots_adjust(wspace=0.5)
    clabel = r'$\eta_t/H\;[\%]$'
    plot_field('eta',result['etar']/abl.H*100,grid,forcing,clabel,-6,6,axarr[0],f)
    clabel = r'$p/\rho_0U_B^2\;[\%]$'
    plot_field('p',result['pr']/abl.Ub**2*100,grid,forcing,clabel,-7,7,axarr[1],f)
    clabel = r'$\Delta u/U\;[\%]$'
    plot_field('M1',M1,grid,forcing,clabel,-20,7,axarr[2],f)
    
    axarr[0].text(-0.15,-0.15,'(a)',transform=axarr[0].transAxes,size=16)
    axarr[1].text(-0.15,-0.15,'(b)',transform=axarr[1].transAxes,size=16)
    axarr[2].text(-0.15,-0.15,'(c)',transform=axarr[2].transAxes,size=16)
    
    plt.savefig('example_figure.png',bbox_inches='tight',dpi=1000)
    
    plt.show()



    
    

        
        
        
        
        
        
    