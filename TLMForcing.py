#!/usr/bin/env python

'''
Forcing models for TLM
'''

__author__ = "Dries Allaerts"
__date__ = "August 7, 2017"

import numpy as np
from scipy import interpolate
from py4sp import mypy
from tlmpy import WakeModel

class CST(object):
    '''
    Constant forcing model
    '''
    def __init__(self,input='default',**kwargs):
        assert input in ['default',
                         'LESbased',
                         'userdefined'], 'Error: CST forcing input mode unknown'

        function = getattr(self,input)
        function(**kwargs)

    def default(self,**kwargs):
        #Default values, corresponding to finWF5
        self.__CT = 0.004
        self.__length = 15000.0
        self.__width = 4800.0

    def LESbased(self,**kwargs):
        assert all([i in kwargs for i in ['sim','abl']]), 'Error: some arguments for LESbased forcing definition are missing'
        
        sim = kwargs['sim']
        abl = kwargs['abl']
        if 'efficiency' in kwargs:
            efficiency = kwargs['efficiency']
        else:
            efficiency=1.0

        CD = (16.0*sim.wf.Ctp)/(4.0+sim.wf.Ctp)**2
        diameter = 2.0*sim.wf.r[0]
        rotorarea = np.pi*sim.wf.r[0]**2
        n = 1.0/(sim.wf.sx*sim.wf.sy*diameter**2)
        areas = mypy.disk_areas(sim.grid.zst,sim.wf.z[0],sim.wf.r[0])
        udisk = np.sum(abl.Ms*areas)/rotorarea
        FT = 0.5*efficiency*CD*udisk**2*rotorarea
        self.__CT = n*FT/abl.S1**2
        self.__length = np.max(sim.wf.x)-np.min(sim.wf.x)+diameter*sim.wf.sx
        self.__width  = np.max(sim.wf.y)-np.min(sim.wf.y)+diameter*sim.wf.sy

    def userdefined(self,**kwargs):
        assert all([i in kwargs for i in ['length','width','CT']]), 'Error: some arguments for userdefined forcing definition are missing'

        self.__length = kwargs['length']
        self.__width  = kwargs['width']
        self.__CT     = kwargs['CT']

    def F0(self,abl,grid):
        F0u = self.CT*abl.S1*abl.U1*self.footprint(grid)
        F0v = self.CT*abl.S1*abl.V1*self.footprint(grid)
        return F0u,F0v

    def F1(self,abl,grid,u1r,v1r):
#        F1u = 2*self.CTr*abl.S1*u1r
#        F1v = 2*self.CTr*abl.S1*v1r
        F1u  = self.CT*(abl.S1+abl.U1**2/abl.S1)*u1r*self.footprint(grid)
        F1u += self.CT*abl.U1*abl.V1/abl.S1*v1r*self.footprint(grid)
        F1v  = self.CT*(abl.S1+abl.V1**2/abl.S1)*v1r*self.footprint(grid)
        F1v += self.CT*abl.U1*abl.V1/abl.S1*u1r*self.footprint(grid)
        return F1u,F1v

    def preprocess(self,abl):
        #No preprocessing neccessary
        pass
            
    @property
    def CT(self):
        return self.__CT
    @property
    def length(self):
        return self.__length
    @property
    def width(self):
        return self.__width

class Stat1Dcst(CST):
    '''
    One-dimensional static constant forcing model
    '''
    def __init__(self,xc,input='default',fringe=None,**kwargs):
        super().__init__(input,**kwargs)
        self.__fringe = fringe
        #Wind farm location
        self.__xstart = xc-self.length/2.0
        self.__xend   = xc+self.length/2.0

    def footprint(self,grid):
        R = mypy.heaviside(grid.xs-self.xstart)-mypy.heaviside(grid.xs-self.xend)
        return R

    def F1fringe(self,grid,u1r,v1r,u2r,v2r):
        #Should only be called when a fringe is present
        F1u1 = self.fringe.Cfringe*u1r*self.fringe.footprint(grid)
        F1v1 = self.fringe.Cfringe*v1r*self.fringe.footprint(grid)
        F1u2 = self.fringe.Cfringe*u2r*self.fringe.footprint(grid)
        F1v2 = self.fringe.Cfringe*v2r*self.fringe.footprint(grid)
        return F1u1,F1v1,F1u2,F1v2

    @property
    def xstart(self):
        return self.__xstart
    @property
    def xend(self):
        return self.__xend
    @property
    def fringe(self):
        return self.__fringe

class Stat2Dcst(CST):
    '''
    Two-dimensional static constant forcing model
    '''
    def __init__(self,xc,yc,input='default',**kwargs):
        super().__init__(input,**kwargs)
        #Wind farm location
        self.__xstart = xc-self.length/2.0
        self.__xend   = xc+self.length/2.0
        self.__ystart = yc-self.width/2.0
        self.__yend   = yc+self.width/2.0
    
    def footprint(self,grid):
        Xs, Ys = np.meshgrid(grid.xs,grid.ys,indexing='ij')
        R = ( (mypy.heaviside(Xs-self.xstart)-mypy.heaviside(Xs-self.xend))*
                (mypy.heaviside(Ys-self.ystart)-mypy.heaviside(Ys-self.yend)) )
        return R

    @property
    def xstart(self):
        return self.__xstart
    @property
    def xend(self):
        return self.__xend
    @property
    def ystart(self):
        return self.__ystart
    @property
    def yend(self):
        return self.__yend

class Dyn1Dcst(Stat1Dcst):
#Depreciated
    '''
    One-dimensional constant forcing model with time-dependent Ct coefficient
    '''

    def __init__(self,grid,input='default',**kwargs):
        super().__init__(grid,input,**kwargs)
        omega = grid.omegas[-1]#20*np.pi/1800.0
        CTtime = np.array([1.0+0.5*np.sin(omega*grid.ts)])
    #    CTtime = np.ones((1,Nt))
        
        self.CTr = np.dot(self.CTr[:,np.newaxis],CTtime)

class Fringe1D(object):
    '''
    Fringe region for one-dimensional simulations
    '''
    def __init__(self,Lf,Cf,method,**kwargs):
        self.__Lfringe = Lf
        self.__Cfringe = Cf
        self.__method  = method
        if self.method=='step':
            #No input arguments required
            self.__drise = None
            self.__dfall = None
        elif self.method=='smoothstep':
            assert all([i in kwargs for i in ['drise','dfall']]), 'Error: missing input arguments for smoothstep fringe'
            self.__drise  = kwargs['drise']
            self.__dfall  = kwargs['dfall']
        else:
            print('Error: fringe method unknown')
            return 1


    def footprint(self,grid):
        #Determine spatial distribution
        if self.method=='step':
            R = mypy.step(grid.xs-(grid.Lx-self.Lfringe))
        elif self.method=='smoothstep':
            xstart = grid.Lx-self.Lfringe
            xend   = grid.Lx
            R = ( mypy.smoothstep((grid.xs-xstart)/self.drise)
                 -mypy.smoothstep((grid.xs-xend)/self.dfall+1) )
        return R

    @property
    def Lfringe(self):
        return self.__Lfringe
    @property
    def Cfringe(self):
        return self.__Cfringe
    @property
    def method(self):
        return self.__method
    @property
    def drise(self):
        return self.__drise
    @property
    def dfall(self):
        return self.__dfall

class WF(object):
    '''
    Wind farm model with individual turbines and Gaussian filtering
    only for 2D grids
    '''
    def __init__(self,xs,ys,diameters,Cts,Lfilter=1000.,wakemodel='nowake',coupling='upstream'):
        self.__Lfilter = Lfilter
        self.__wakemodel = wakemodel
        self.__coupling  = coupling
        self.__Nturb = len(xs)
        self.__xstart = np.min(xs)
        self.__xend   = np.max(xs)
#        self.__length = self.xend-self.xstart
        self.__ystart = np.min(ys)
        self.__yend   = np.max(ys)
#        self.__width  = self.yend-self.ystart
        self.__St = None
        self.__Stjac = None
        self.__turbines = []
        self.initturbines(xs,ys,diameters,Cts)

    def initturbines(self,xs,ys,diameters,Cts):
        #List with all turbines
        for turb in range(self.Nturb):
            self.__turbines.append(turbine(xs[turb],ys[turb],
                                   diameters[turb],Cts[turb]))


    def preprocess(self,abl):
        #Compute FT and Ftjac matrices in preprocessing as this might take a while
        function = getattr(WakeModel,self.wakemodel)
        self.__St = function(self.turbines,abl)
        function = getattr(WakeModel,self.wakemodel+'_jac')
        self.__Stjac = function(self.turbines,abl)
    
    def F0(self,abl,grid):
        #Filter turbine forces onto grid
        F0u = np.zeros(grid.shape)
        F0v = np.zeros(grid.shape)
        e_str = WakeModel.e_streamwise(abl.U1,abl.V1)
        for index,turb in enumerate(self.turbines):
            F0u += (0.5 * turb.Ct * turb.rotorarea *
                    self.St[index]**2 * e_str[0] *
                    turb.footprint(grid,self.Lfilter) )
            F0v += (0.5 * turb.Ct * turb.rotorarea *
                    self.St[index]**2 * e_str[1] *
                    turb.footprint(grid,self.Lfilter) )
        return F0u,F0v

    def F1(self,abl,grid,u1r,v1r):
        #Get velocity perturbations
        function = getattr(self,'u_'+self.coupling)
        u1inf,v1inf = function(abl,grid,u1r,v1r)
        #Unit vectors and their derivatives
        e_str  = WakeModel.e_streamwise(abl.U1,abl.V1)
        E_str  = WakeModel.e_str_jac(abl.U1,abl.V1)
        #Filter turbine forces onto grid
        F1u = np.zeros(grid.shape)
        F1v = np.zeros(grid.shape)
        for index,turb in enumerate(self.turbines):
            F1u += (0.5 * turb.Ct * turb.rotorarea *
                        ( 2*self.St[index]*self.Stjac[index,0]*e_str[0] +
                            self.St[index]**2*E_str[0,0] ) *
                    u1inf*turb.footprint(grid,self.Lfilter) )
            F1u += (0.5 * turb.Ct * turb.rotorarea *
                        ( 2*self.St[index]*self.Stjac[index,1]*e_str[0] +
                            self.St[index]**2*E_str[0,1] ) *
                    v1inf*turb.footprint(grid,self.Lfilter) )
            F1v += (0.5 * turb.Ct * turb.rotorarea *
                        ( 2*self.St[index]*self.Stjac[index,0]*e_str[1] +
                            self.St[index]**2*E_str[1,0] ) *
                    u1inf*turb.footprint(grid,self.Lfilter) )
            F1v += (0.5 * turb.Ct * turb.rotorarea *
                        ( 2*self.St[index]*self.Stjac[index,1]*e_str[1] +
                            self.St[index]**2*E_str[1,1] ) *
                    v1inf*turb.footprint(grid,self.Lfilter) )
        return F1u,F1v

    def P0(self,abl,grid):
        P0 = np.zeros(grid.shape)
        for index,turb in enumerate(self.turbines):
            P0 += (0.5 * turb.Cp * turb.rotorarea * self.St[index]**3 *
                   turb.footprint(grid,self.Lfilter) )
        return P0

    def P1(self,abl,grid,u1r,v1r):
        #Get velocity perturbations
        function = getattr(self,'u_'+self.coupling)
        u1inf,v1inf = function(abl,grid,u1r,v1r)
        #Filter turbine power onto grid
        P1 = np.zeros(grid.shape)
        for index,turb in enumerate(self.turbines):
            P1 += (0.5 * turb.Cp * turb.rotorarea *
                        ( 3*self.St[index]**2*self.Stjac[index,0] ) *
                    u1inf*turb.footprint(grid,self.Lfilter) )
            P1 += (0.5 * turb.Cp * turb.rotorarea *
                        ( 3*self.St[index]**2*self.Stjac[index,1] ) *
                    v1inf*turb.footprint(grid,self.Lfilter) )
        return P1

    def Ftot0(self,abl,grid):
        '''Total Wind-farm force without linear correction'''
        F0u,F0v = self.F0(abl,grid)
        F = np.sqrt(F0u**2+F0v**2)
        return np.sum(F)*grid.dx*grid.dy

    def Ftot(self,abl,grid,u1r,v1r):
        '''Total Wind-farm force with linear correction'''
        F0u,F0v = self.F0(abl,grid)
        F1u,F1v = self.F1(abl,grid,u1r,v1r)
        F = np.sqrt( (F0u+F1u)**2+(F0v+F1v)**2 )
        return np.sum(F)*grid.dx*grid.dy

    def Ftheory(self,abl,grid):
        '''Theoretical Wind-farm force without gravity waves or wake effects'''
        F = 0.
        for index,turb in enumerate(self.turbines):
            F += 0.5 * turb.Ct * turb.rotorarea * abl.S1**2
        return F

    def Ptot0(self,abl,grid):
        '''Total Wind-farm power without linear correction'''
        P0 = self.P0(abl,grid)
        return np.sum(P0)*grid.dx*grid.dy

    def Ptot(self,abl,grid,u1r,v1r):
        '''Total Wind-farm power with linear correction'''
        P0 = self.P0(abl,grid)
        P1 = self.P1(abl,grid,u1r,v1r)
        P = P0 + P1
        return np.sum(P)*grid.dx*grid.dy

    def Ptheory(self,abl,grid):
        '''Theoretical Wind-farm power without gravity waves or wake effects'''
        P = 0.
        for index,turb in enumerate(self.turbines):
            P += 0.5 * turb.Cp * turb.rotorarea * abl.S1**3
        return P

    def u_upstream(self,abl,grid,u1r,v1r):
        #Coupling based on velocity 10D upstream from the first turbine
        WDvector = np.array([abl.U1/abl.S1,abl.V1/abl.S1])
        index = self.firstTurbine(WDvector)
        xloc = self.turbines[index].x-10*self.turbines[index].D*WDvector[0]
        yloc = self.turbines[index].y-10*self.turbines[index].D*WDvector[1]
        fu = interpolate.interp2d(grid.xs,grid.ys,u1r.T)
        fv = interpolate.interp2d(grid.xs,grid.ys,v1r.T)
        u1inf = np.asscalar(fu(xloc,yloc))
        v1inf = np.asscalar(fv(xloc,yloc))
        return u1inf,v1inf

    def u_farm(self,abl,grid,u1r,v1r):
        #Coupling based on farm-averaged velocity
        fu = interpolate.interp2d(grid.xs,grid.ys,u1r.T)
        fv = interpolate.interp2d(grid.xs,grid.ys,v1r.T)
        xs = np.linspace(self.xstart,self.xend,int(self.length/grid.dx))
        ys = np.linspace(self.ystart,self.yend,int(self.width/grid.dy))
        X,Y = np.meshgrid(xs,ys,indexing='ij')
        u1inf = np.mean(fu(np.ravel(X),np.ravel(Y)))
        v1inf = np.mean(fv(np.ravel(X),np.ravel(Y)))
        return u1inf,v1inf
        

    def firstTurbine(self,WDvector):
        #Find first turbine in a given wind direction by projecting the
        #coordinates on the wind direction vector. The minimal distance
        #corresponds to the first turbine
        x = np.array([self.turbines[i].x for i in range(self.Nturb)])
        y = np.array([self.turbines[i].y for i in range(self.Nturb)])
        coordinates = np.concatenate([x,y]).reshape(self.Nturb,2,order='F')
        dist = np.dot(coordinates,WDvector)
        return np.argmin(dist)

    @property
    def Lfilter(self):
        return self.__Lfilter
    @property
    def wakemodel(self):
        return self.__wakemodel
    @property
    def coupling(self):
        return self.__coupling
    @property
    def Nturb(self):
        return self.__Nturb
    @property
    def turbines(self):
        return self.__turbines
    @property
    def St(self):
        return self.__St
    @property
    def Stjac(self):
        return self.__Stjac
    @property
    def length(self):
        return self.xend-self.xstart
#        return self.__length
#    @length.setter
#    def length(self,value):
#        self.__length = value
    @property
    def width(self):
        return self.yend-self.ystart
#        return self.__width
#    @width.setter
#    def width(self,value):
#        self.__width = value
    @property
    def xstart(self):
        return self.__xstart
    @property
    def xend(self):
        return self.__xend
    @property
    def ystart(self):
        return self.__ystart
    @property
    def yend(self):
        return self.__yend

class WF1D(WF):
    '''
    Wind farm model on 1D grid with individual turbines and Gaussian filtering
    '''
    def __init__(self,xs,ys,diameters,Cts,Lfilter=1000.,wakemodel='nowake',coupling='upstream'):
        super().__init__(xs,ys,diameters,Cts,Lfilter,wakemodel,coupling)
        self.__fringe = None

    def initturbines(self,xs,ys,diameters,Cts):
        yc = self.ystart+self.width/2.0
        for turb in range(self.Nturb):
            #Implementation is a bit at hoc: __turbines attribute is from
            #parent class but private attributes are not inherited
            self._WF__turbines.append(turbine1D(xs[turb],ys[turb]-yc,
                                   diameters[turb],Cts[turb]))

    def u_upstream(self,abl,grid,u1r,v1r):
        #Coupling based on velocity 10D upstream from the first turbine
        index = self.firstTurbine()
        xloc = self.turbines[index].x-10*self.turbines[index].D
        fu = interpolate.interp1d(grid.xs,u1r)
        fv = interpolate.interp1d(grid.xs,v1r)
        u1inf = np.asscalar(fu(xloc))
        v1inf = np.asscalar(fv(xloc))
        return u1inf,v1inf

    def u_farm(self,abl,grid,u1r,v1r):
        #Coupling based on farm-averaged velocity
        fu = interpolate.interp1d(grid.xs,u1r)
        fv = interpolate.interp1d(grid.xs,v1r)
        xs = np.linspace(self.xstart,self.xend,int(self.length/grid.dx))
        u1inf = np.mean(fu(xs))
        v1inf = np.mean(fv(xs))
        return u1inf,v1inf

    def firstTurbine(self):
        #Find first turbine (along x direction)
        x = np.array([self.turbines[i].x for i in range(self.Nturb)])
        return np.argmin(x)

    @property
    def fringe(self):
        return self.__fringe

class turbine(object):
    '''
    Turbine object
    only for 2D grids
    '''
    def __init__(self,xloc,yloc,diameter,thrustcoefficient):
        self.__x = xloc
        self.__y = yloc
        self.__D = diameter
        self.__Ct = thrustcoefficient

    def footprint(self,grid,L):
        Xs, Ys = np.meshgrid(grid.xs,grid.ys,indexing='ij')
        dist = (Xs-self.x)**2+(Ys-self.y)**2
        R = 1./(np.pi*L**2)*np.exp(-dist/L**2)
        return R

    @property
    def x(self):
        return self.__x
    @property
    def y(self):
        return self.__y
    @property
    def D(self):
        return self.__D
    @property
    def rotorarea(self):
        return np.pi/4.0*self.D**2
    @property
    def Ct(self):
        return self.__Ct
    @property
    def induction(self):
        return 0.5-0.5*np.sqrt(1-self.Ct)
    @property
    def Cp(self):
        return 4*self.induction*(1-self.induction)**2

class turbine1D(turbine):
    '''
    Special 1D turbine object
    (footprint is a line at y=0)
    '''
    def footprint(self,grid,L):
        dist = (grid.xs-self.x)**2+(self.y)**2
        R = 1./(np.pi*L**2)*np.exp(-dist/L**2)
        return R
