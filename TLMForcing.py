#!/usr/bin/env python

'''
Forcing models for TLM
'''

import numpy as np
from scipy import interpolate
from py4sp import mypy
from tlmpy import WakeModel
from tlmpy import TLM_tools

class CST(object):
    '''
    Common class for perturbing force models with a constant drag coefficient
    '''
    def __init__(self,input='default',**kwargs):
        '''
        Initialise forcing parameters with one of the following valid methods:
        - default forcing parameters
        - based on LES data
        - directly specify forcing parameters

        Parameters
        ----------
        input (optional): str
            Name of the method used to specify the forcing parameters
            Default: 'default' method (use default forcing parameters)
        '''
        assert input in ['default',
                         'LESbased',
                         'userdefined'], 'Error: CST forcing input mode unknown'

        function = getattr(self,input)
        function(**kwargs)

    def default(self,**kwargs):
        '''
        Method to specify the forcing parameters as the default implemented values,
        which correspond to the wind-farm parameters of case S1 of
        Allaerts and Meyers, J. Fluid Mech. 814, 2017.
        '''
        self.__CT = 0.004
        self.__length = 15000.0
        self.__width = 4800.0

    def LESbased(self,**kwargs):
        '''
        Method to specify the forcing parameters based on LES data

        This routine assimes that the LES data is obtained with SP-Wind, and
        the input parameters include a specific data structure of the SP-Wind
        post-processing tools

        Parameters
        ----------
        sim: Simulation object (defined in simulation.py of py4sp package)
            LES simulation meta data structure
        abl: ABL object (defined in TLM.py or SLM.py)
            atmospheric state of the Three-layer model
        '''
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
        '''
        Method to specify the forcing parameters directly

        Parameters
        ----------
        length,width: float
            Length and width of the area where the perturbing force is added
        CT: float
            Drag coefficient of the perturbing force
        '''
        assert all([i in kwargs for i in ['length','width','CT']]), 'Error: some arguments for userdefined forcing definition are missing'

        self.__length = kwargs['length']
        self.__width  = kwargs['width']
        self.__CT     = kwargs['CT']

    def F0(self,abl,grid):
        '''
        Compute the zero-th order term of the drag force on the given numerical grid

        Parameters
        ----------
        abl: ABL object (defined in TLM.py or SLM.py)
            Atmospheric state
        grid: Grid object (defined in TLM.py)
            Numerical grid

        Returns
        -------
        F0u, F0v: numpy array with same shape as grid
            Zero-th order term of the drag force (real space)
        '''
        F0u = self.CT*abl.S1*abl.U1*self.footprint(grid)
        F0v = self.CT*abl.S1*abl.V1*self.footprint(grid)
        return F0u,F0v

    def F1(self,abl,grid,u1r,v1r):
        '''
        Compute the first order term of the drag force on the given numerical grid

        Parameters
        ----------
        abl: ABL object (defined in TLM.py or SLM.py)
            Atmospheric state
        grid: Grid object (defined in TLM.py)
            Numerical grid
        u1r,v1r: numpy array with same shape as grid
            Perturbation velocities

        Returns
        -------
        F1u, F1v: numpy array with same shape as grid
            First order term of the drag force (real space)
        '''
        F1u  = self.CT*(abl.S1+abl.U1**2/abl.S1)*u1r*self.footprint(grid)
        F1u += self.CT*abl.U1*abl.V1/abl.S1*v1r*self.footprint(grid)
        F1v  = self.CT*(abl.S1+abl.V1**2/abl.S1)*v1r*self.footprint(grid)
        F1v += self.CT*abl.U1*abl.V1/abl.S1*u1r*self.footprint(grid)
        return F1u,F1v

    def preprocess(self,abl):
        '''
        Standard function for preprocessing

        Constant drag coefficient models do not require preprocessing,
        so this function is empty
        '''
        pass
            
    @property
    def CT(self):
        '''Drag coefficient'''
        return self.__CT
    @property
    def length(self):
        '''Length of the area where the perturbing force is added'''
        return self.__length
    @property
    def width(self):
        '''Width of the area where the perturbing force is added'''
        return self.__width

class Stat1Dcst(CST):
    '''
    Constant drag coefficient model for static one-dimensional simulations
    '''
    def __init__(self,xc,input='default',fringe=None,**kwargs):
        '''
        Initialise forcing parameters with one of the following valid methods:
        - default forcing parameters
        - based on LES data
        - directly specify forcing parameters

        Parameters
        ----------
        xc: float
            x coordinate of the center of the forcing region
        input (optional): str
            Name of the method used to specify the forcing parameters
            Default: 'default' method (use default forcing parameters)
        fringe: Fringe1D object
            fringe region forcing model
        '''
        super().__init__(input,**kwargs)
        self.__fringe = fringe
        #Wind farm location
        self.__xstart = xc-self.length/2.0
        self.__xend   = xc+self.length/2.0

    def footprint(self,grid):
        '''
        Compute the geometrical footprint of the perturbing force for the given grid

        Parameters
        ----------
        grid: Grid object (defined in TLM.py)
            Numerical grid

        Returns
        -------
        R: numpy array with same shape as grid
            Geometrical footprint of the perturbing force
        '''
        R = mypy.heaviside(grid.xs-self.xstart)-mypy.heaviside(grid.xs-self.xend)
        return R

    def F1fringe(self,grid,u1r,v1r,u2r,v2r):
        '''
        Compute the fringe forcing
        
        Should only be called when a fringe is present

        Parameters
        ----------
        grid: Grid object (defined in TLM.py)
            Numerical grid
        u1r,v1r,u2r,v2r: numpy array with same shape as grid
            Perturbation velocities in the wind-farm and upper layer

        Returns
        -------
        F1u1,F1v1,F1u2,F1v2: numpy array with same shape as grid
            Fringe forcing for x and y momentum in wind-farm and upper layer
            (real space)
        '''
        F1u1 = self.fringe.Cfringe*u1r*self.fringe.footprint(grid)
        F1v1 = self.fringe.Cfringe*v1r*self.fringe.footprint(grid)
        F1u2 = self.fringe.Cfringe*u2r*self.fringe.footprint(grid)
        F1v2 = self.fringe.Cfringe*v2r*self.fringe.footprint(grid)
        return F1u1,F1v1,F1u2,F1v2

    @property
    def xstart(self):
        '''x coordinate of the start of the perturbing force'''
        return self.__xstart
    @property
    def xend(self):
        '''x coordinate of the end of the perturbing force'''
        return self.__xend
    @property
    def fringe(self):
        '''fringe region forcing model'''
        return self.__fringe

class Stat2Dcst(CST):
    '''
    Constant drag coefficient model for static two-dimensional simulations
    '''
    def __init__(self,xc,yc,input='default',**kwargs):
        '''
        Initialise forcing parameters with one of the following valid methods:
        - default forcing parameters
        - based on LES data
        - directly specify forcing parameters

        Parameters
        ----------
        xc,yc: float
            x and y coordinate of the center of the forcing region
        input (optional): str
            Name of the method used to specify the forcing parameters
            Default: 'default' method (use default forcing parameters)
        '''
        super().__init__(input,**kwargs)
        #Wind farm location
        self.__xstart = xc-self.length/2.0
        self.__xend   = xc+self.length/2.0
        self.__ystart = yc-self.width/2.0
        self.__yend   = yc+self.width/2.0
    
    def footprint(self,grid):
        '''
        Compute the geometrical footprint of the perturbing force for the given grid

        Parameters
        ----------
        grid: Grid object (defined in TLM.py)
            Numerical grid

        Returns
        -------
        R: numpy array with same shape as grid
            Geometrical footprint of the perturbing force
        '''
        Xs, Ys = np.meshgrid(grid.xs,grid.ys,indexing='ij')
        R = ( (mypy.heaviside(Xs-self.xstart)-mypy.heaviside(Xs-self.xend))*
                (mypy.heaviside(Ys-self.ystart)-mypy.heaviside(Ys-self.yend)) )
        return R

    @property
    def xstart(self):
        '''x coordinate of the start of the perturbing force'''
        return self.__xstart
    @property
    def xend(self):
        '''x coordinate of the end of the perturbing force'''
        return self.__xend
    @property
    def ystart(self):
        '''y coordinate of the start of the perturbing force'''
        return self.__ystart
    @property
    def yend(self):
        '''y coordinate of the end of the perturbing force'''
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
    Fringe region forcing model for one-dimensional simulations
    '''
    def __init__(self,Lf,Cf,method,**kwargs):
        '''
        Initialise fringe parameters with either a step or smooth step function

        **kwargs must contain drise and dfall in case smooth step function is used

        Parameters
        ----------
        Lf: float
            Length of the fringe region
        Cf: float
            Drag coefficient
        method: str
            Form of the geometric footprint of the fringe region
            (step or smooth step)
        '''
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
        '''
        Compute the geometrical footprint of the fringe region forcing
        for the given grid

        Parameters
        ----------
        grid: Grid object (defined in TLM.py)
            Numerical grid

        Returns
        -------
        R: numpy array with same shape as grid
            Geometrical footprint of the fringe region force
        '''
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
        '''Fringe region length'''
        return self.__Lfringe
    @property
    def Cfringe(self):
        '''Fringe region drag coefficient'''
        return self.__Cfringe
    @property
    def method(self):
        '''Form of the geometric footprint of the fringe region'''
        return self.__method
    @property
    def drise(self):
        '''
        Length over which the fringe region force rises from zero to its maximum
        value in case of a smooth step form
        '''
        return self.__drise
    @property
    def dfall(self):
        '''
        Length over which the fringe region force falls from its maximum value
        to zero in case of a smooth step form
        '''
        return self.__dfall

class WF(object):
    '''
    Wind farm object

    Wind-farm perturbing force model with individual turbines and Gaussian filtering
    only for 2D grids
    '''
    def __init__(self,grid,xs,ys,diameters,Cts,zhs,Lfilter=1000.,wakemodel='nowake',coupling='upstream'):
        '''
        Initialise the wind farm model with choice of wake model and coupling
        wake model:
        - 'nowake': no wake effects
        - 'jensen': Jensen wake model (depreciated)
        - 'gauss' : Gaussian wake model

        coupling:
        - 'upstream': coupling based on velocity 10D upstream of the first turbine
        - 'farm'    : coupling based on farm-averaged velocity

        Parameters
        ----------
        grid: Grid object
            numerical grid
        xs,ys: 1d numpy array
            x and y location of the individual turbines
        diameters, Cts, zhs: 1d numpy array
            diameters, thrust coefficients and hub height of individual turbines
        Lfilter (optional): float
            filter length for the Gaussian filter
            default: 1000.
        wakemodel (optional): str
            name of the wake model used to compute individual turbine thrust force
            default: nowake
        coupling (optional): str
            name of the coupling method
            default: upstream
        '''
        self.__grid = grid
        self.__xs = xs
        self.__ys = ys
        self.__Lfilter = Lfilter
        self.__wakemodel = wakemodel
        self.__coupling  = coupling
        self.__Nturb = len(xs)
        self.__xstart = np.min(xs)
        self.__xend   = np.max(xs)
        self.__ystart = np.min(ys)
        self.__yend   = np.max(ys)
        self.__St = None
        self.__Stjac = None
        self.__Footprint = [0] * len(xs)          
        self.__Footprint32 = [0] * len(xs)        
        self.__turbines = []
        self.initturbines(xs,ys,diameters,Cts,zhs)
        

    def initturbines(self,xs,ys,diameters,Cts,zhs):
        '''
        Make list with turbines
        
        Parameters
        ----------
        xs,ys: 1d numpy array
            x and y location of the individual turbines
        diameters, Cts, zhs: 1d numpy array
            diameters, thrust coefficients and hub height of individual turbines
        '''
        #List with all turbines
        for turb in range(self.Nturb):
            self.__turbines.append(turbine(xs[turb],ys[turb],
                                   diameters[turb],Cts[turb],zhs[turb]))

    
    def preprocess(self,abl,grid,grid32,WFfeedback=True):
        '''
        Standard function for preprocessing
        
        Compute inflow velocities St and Jacobian of inflow velocities Stjac
        in preprocessing step as this is independent of the perturbation velocity.
        The footprint is a list of sparse matrix.

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object
            numerical grid
        grid32: Grid object
            dealiasing grid
        WFfeedback (optional): bool
            flag to turn on/off two-way coupling of wind-farm forcing
            default: True
        '''
        function = getattr(WakeModel,self.wakemodel)
        self.__St,self.__Stjac = function(self.turbines,abl)
        if WFfeedback==False:
            self.__Stjac = np.zeros((self.Nturb,2))
        function = getattr(WakeModel,'footprint')
        for index,turb in enumerate(self.turbines):
            self.__Footprint[index]   = function(grid.xs,grid.ys,grid.dx,grid.dy,self.Lfilter,
                                                    self.xs[index],self.ys[index])
            self.__Footprint32[index] = function(grid32.xs,grid32.ys,grid32.dx,grid32.dy,self.Lfilter,
                                                    self.xs[index],self.ys[index])
    
    def F0(self,abl,grid):
        '''
        Compute zero-th order wind-farm force on specified grid

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        F0u, F0v: 2d numpy array
            zero-th order wind-farm force in x and y (same shape as grid)
        '''
        Nx = grid.Nx
        Ny = grid.Ny
        F0 = np.zeros((2,Nx,Ny))

        e_str = WakeModel.e_streamwise(abl.U1,abl.V1)
        Ct = [self.turbines[i].Ct for i in range(self.Nturb)]
        rotorarea = [self.turbines[i].rotorarea for i in range(self.Nturb)]
        
        for index in range(self.Nturb): 
            F0 = TLM_tools.evaluate_F0(e_str,F0[0],F0[1],
                                            Ct[index],rotorarea[index],self.St[index],
                                            self.Footprint[index][0],
                                            self.Footprint[index][1],
                                            self.Footprint[index][2]) 
        return F0[0],F0[1]
    
    def F1(self,abl,grid,u1r,v1r):
        '''
        Compute first order wind-farm force on specified grid for given
        perturbation velocities

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        F1u,F1v: 2d numpy array
            first order wind-farm force in x and y (same shape as grid)
        '''
        #Get velocity perturbations
        function = getattr(self,'u_'+self.coupling)
        u1inf,v1inf = function(abl,grid,u1r,v1r)
        #Unit vectors and their derivatives
        Nx = grid.Nx
        Ny = grid.Ny
        F1 = np.zeros((2,Nx,Ny))
        
        e_str  = WakeModel.e_streamwise(abl.U1,abl.V1)
        E_str  = WakeModel.e_str_jac(abl.U1,abl.V1)
        Ct = [self.turbines[i].Ct for i in range(self.Nturb)]
        rotorarea = [self.turbines[i].rotorarea for i in range(self.Nturb)]
        
        for index in range(self.Nturb):      
            F1 = TLM_tools.evaluate_F1(e_str,E_str,F1[0],F1[1],
                                             Ct[index],rotorarea[index],self.St[index],
                                             self.Stjac[index,0],self.Stjac[index,1],
                                             self.Footprint32[index][0],
                                             self.Footprint32[index][1],
                                             self.Footprint32[index][2],
                                             u1inf,v1inf)        
        return F1[0],F1[1] 
    
    def P0(self,abl,grid):
        '''
        Compute zero-th order wind-farm power (on specified grid)

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        P0: 2d numpy array
            zero-th order wind-farm power (same shape as grid)
        '''
        P0 = np.zeros(grid.shape)
        
        Cp = [self.turbines[i].Cp for i in range(self.Nturb)]
        rotorarea = [self.turbines[i].rotorarea for i in range(self.Nturb)]
        
        for index in range(self.Nturb):
            P0 = TLM_tools.evaluate_P0(P0,Cp[index],rotorarea[index],self.St[index],
                                       self.Footprint[index][0],
                                       self.Footprint[index][1],
                                       self.Footprint[index][2])
        return P0
    
    
    def P1(self,abl,grid,u1r,v1r):
        '''
        Compute first order wind-farm power (on specified grid) for given
        perturbation velocities

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        P1: 2d numpy array
            first order wind-farm power (same shape as grid)
        '''
        #Get velocity perturbations
        function = getattr(self,'u_'+self.coupling)
        u1inf,v1inf = function(abl,grid,u1r,v1r)
        #Filter turbine power onto grid
        P1 = np.zeros(grid.shape)
        
        Cp = [self.turbines[i].Cp for i in range(self.Nturb)]
        rotorarea = [self.turbines[i].rotorarea for i in range(self.Nturb)]
        
        for index in range(self.Nturb):
            P1 = TLM_tools.evaluate_P1(P1,Cp[index],rotorarea[index],self.St[index],
                                        self.Stjac[index,0],self.Stjac[index,1],
                                        self.Footprint[index][0],
                                        self.Footprint[index][1],
                                        self.Footprint[index][2],
                                        u1inf,v1inf)

        return P1

    def Ftot0(self,abl,grid):
        '''
        Total wind-farm force without linear correction
        (hence no two-way coupling and no gravity-wave feedback effects)

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        _: float
            Total wind-farm force without linear correction
        '''
        F0u,F0v = self.F0(abl,grid)
        F = np.sqrt(F0u**2+F0v**2)
        return np.sum(F)*grid.dx*grid.dy

    def Ftot(self,abl,grid,u1r,v1r):
        '''
        Total wind-farm force with linear correction

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        _: float
            Total wind-farm force with linear correction
        '''
        F0u,F0v = self.F0(abl,grid)
        F1u,F1v = self.F1(abl,grid,u1r,v1r)
        F = np.sqrt( (F0u+F1u)**2+(F0v+F1v)**2 )
        return np.sum(F)*grid.dx*grid.dy

    def Ftheory(self,abl,grid):
        '''
        Theoretical wind-farm force without gravity waves or wake effects

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        _: float
            Theoretical wind-farm force without gravity waves or wake effects
        '''
        F = 0.
        for index,turb in enumerate(self.turbines):
            F += 0.5 * turb.Ct * turb.rotorarea * abl.S1**2
        return F

    def Ptot0(self,abl,grid):
        '''
        Total wind-farm power without linear correction
        (hence no two-way coupling and no gravity-wave feedback effects)

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        _: float
            Total wind-farm power without linear correction
        '''
        P0 = self.P0(abl,grid)
        return np.sum(P0)*grid.dx*grid.dy

    def Ptot(self,abl,grid,u1r,v1r):
        '''
        Total wind-farm power with linear correction

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        _: float
            Total wind-farm power with linear correction
        '''
        P0 = self.P0(abl,grid)
        P1 = self.P1(abl,grid,u1r,v1r)
        P = P0 + P1
        return np.sum(P)*grid.dx*grid.dy

    def Ptheory(self,abl,grid):
        '''
        Theoretical wind-farm power without gravity waves or wake effects

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid

        Returns
        -------
        _: float
            Theoretical wind-farm power without gravity waves or wake effects
        '''
        P = 0.
        for index,turb in enumerate(self.turbines):
            P += 0.5 * turb.Cp * turb.rotorarea * abl.S1**3
        return P

    def u_upstream(self,abl,grid,u1r,v1r):
        '''
        Calculate perturbation velocity for wake model based on upstream coupling
        strategy, i.e., based on velocity 10D upstream of the first turbine

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        u1inf,v1inf: float
            perturbation velocity in x and y for wake model
        '''
        WDvector = np.array([abl.U1/abl.S1,abl.V1/abl.S1])
        index = self.firstTurbine(WDvector)        
        xloc = self.turbines[index].x-10*self.turbines[index].D*WDvector[0]
        yloc = self.turbines[index].y-10*self.turbines[index].D*WDvector[1]
        
        #To save time, the interpolatation is done only over a 6x6 grid centered in xloc,yloc                
        limit = 3
        start_x = int(xloc/grid.dx-limit)
        end_x = int(xloc/grid.dx+limit)
        start_y = int(yloc/grid.dy-limit)
        end_y = int(yloc/grid.dy+limit)
        
        fu = interpolate.interp2d(grid.xs[start_x:end_x],grid.ys[start_y:end_y],u1r[start_x:end_x,start_y:end_y].T)
        fv = interpolate.interp2d(grid.xs[start_x:end_x],grid.ys[start_y:end_y],v1r[start_x:end_x,start_y:end_y].T)
        u1inf = np.asscalar(fu(xloc,yloc))
        v1inf = np.asscalar(fv(xloc,yloc))
        return u1inf,v1inf
    
    def u_farm(self,abl,grid,u1r,v1r):
        '''
        Calculate perturbation velocity for wake model based on farm-averaged
        coupling strategy, i.e., based on farm-averaged velocity

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 2d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        u1inf,v1inf: float
            perturbation velocity in x and y for wake model
        '''
        fu = interpolate.interp2d(grid.xs,grid.ys,u1r.T)
        fv = interpolate.interp2d(grid.xs,grid.ys,v1r.T)
        xs = np.linspace(self.xstart,self.xend,int(self.length/grid.dx))
        ys = np.linspace(self.ystart,self.yend,int(self.width/grid.dy))
        X,Y = np.meshgrid(xs,ys,indexing='ij')
        u1inf = np.mean(fu(np.ravel(X),np.ravel(Y)))
        v1inf = np.mean(fv(np.ravel(X),np.ravel(Y)))
        return u1inf,v1inf

    def firstTurbine(self,WDvector):
        '''
        Find first turbine for a given wind direction

        Parameters
        ----------
        WDvector: 1d numpy array with size 2
            wind direction vector

        Returns
        -------
        _: integer
            index of the first turbine in the given wind direction
        '''
        #Find first turbine in a given wind direction by projecting the
        #coordinates on the wind direction vector. The minimal distance
        #corresponds to the first turbine
        x = np.array([self.turbines[i].x for i in range(self.Nturb)])
        y = np.array([self.turbines[i].y for i in range(self.Nturb)])
        coordinates = np.concatenate([x,y]).reshape(self.Nturb,2,order='F')
        dist = np.dot(coordinates,WDvector)
        return np.argmin(dist)
 
    @property
    def grid(self):
        '''Numerical grid'''
        return self.__grid
    @property
    def xs(self):
        '''wind turbine x-coordinate'''
        return self.__xs
    @property
    def ys(self):
        '''wind turbine y-coordinate'''
        return self.__ys
    @property
    def Lfilter(self):
        '''Gaussian filter length'''
        return self.__Lfilter
    @property
    def wakemodel(self):
        '''Type of wake model'''
        return self.__wakemodel
    @property
    def coupling(self):
        '''Coupling strategy'''
        return self.__coupling
    @property
    def Nturb(self):
        '''Number of turbines'''
        return self.__Nturb
    @property
    def turbines(self):
        '''List of turbine objects'''
        return self.__turbines
    @property
    def St(self):
        '''Inflow velocities'''
        return self.__St
    @property
    def Stjac(self):
        '''Jacobian of inflow velocities'''
        return self.__Stjac
    @property
    def Footprint(self):
        '''Wind turbine footprint'''
        return self.__Footprint
    @property
    def Footprint32(self):
        '''Wind turbine footprint on dealiasing grid'''
        return self.__Footprint32
    @property
    def dAdu(self):
        '''U-derivative of System matrix of turbine inflow velocities linear system'''
        return self.__dAdu
    @property
    def dAdv(self):
        '''V-derivative of System matrix of turbine inflow velocities linear system'''
        return self.__dAdv
    @property
    def length(self):
        '''Length of area covered by the turbines'''
        return self.xend-self.xstart
    @property
    def width(self):
        '''Width of area covered by the turbines'''
        return self.yend-self.ystart
    @property
    def xstart(self):
        '''x coordinate of the start of the wind-farm area'''
        return self.__xstart
    @property
    def xend(self):
        '''x coordinate of the end of the wind-farm area'''
        return self.__xend
    @property
    def ystart(self):
        '''y coordinate of the start of the wind-farm area'''
        return self.__ystart
    @property
    def yend(self):
        '''y coordinate of the end of the wind-farm area'''
        return self.__yend
    @property
    def xcentre(self):
        '''x coordinate of the centre of the wind-farm area'''
        return (self.xstart+self.xend)/2.0
    @property
    def ycentre(self):
        '''y coordinate of the centre of the wind-farm area'''
        return (self.ystart+self.yend)/2.0

class WF1D(WF):
    '''
    Wind farm object (1D alternative)

    Wind-farm perturbing force model with individual turbines and Gaussian
    filtering for 1D grids
    '''
    def __init__(self,xs,ys,diameters,Cts,Lfilter=1000.,wakemodel='nowake',coupling='upstream'):
        '''
        Initialise the wind farm model with choice of wake model and coupling
        wake model:
        - 'nowake': no wake effects
        - 'jensen': Jensen wake model (depreciated)
        - 'gauss' : Gaussian wake model

        coupling:
        - 'upstream': coupling based on velocity 10D upstream of the first turbine
        - 'farm'    : coupling based on farm-averaged velocity

        Parameters
        ----------
        xs,ys: 1d numpy array
            x and y location of the individual turbines
        diameters, Cts: 1d numpy array
            diameters and thrust coefficients of individual turbines
        Lfilter (optional): float
            filter length for the Gaussian filter
            default: 1000.
        wakemodel (optional): str
            name of the wake model used to compute individual turbine thrust force
            default: nowake
        coupling (optional): str
            name of the coupling method
            default: upstream
        '''
        super().__init__(xs,ys,diameters,Cts,Lfilter,wakemodel,coupling)
        self.__fringe = None

    def initturbines(self,xs,ys,diameters,Cts):
        '''
        Make list with turbines
        
        Parameters
        ----------
        xs,ys: 1d numpy array
            x and y location of the individual turbines
        diameters, Cts: 1d numpy array
            diameters and thrust coefficients of individual turbines
        '''
        yc = self.ystart+self.width/2.0
        for turb in range(self.Nturb):
            #Implementation is a bit at hoc: __turbines attribute is from
            #parent class but private attributes are not inherited
            self._WF__turbines.append(turbine1D(xs[turb],ys[turb]-yc,
                                   diameters[turb],Cts[turb]))

    def u_upstream(self,abl,grid,u1r,v1r):
        '''
        Calculate perturbation velocity for wake model based on upstream coupling
        strategy, i.e., based on velocity 10D upstream of the first turbine

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 1d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        u1inf,v1inf: float
            perturbation velocity in x and y for wake model
        '''
        index = self.firstTurbine()
        xloc = self.turbines[index].x-10*self.turbines[index].D
        fu = interpolate.interp1d(grid.xs,u1r)
        fv = interpolate.interp1d(grid.xs,v1r)
        u1inf = np.asscalar(fu(xloc))
        v1inf = np.asscalar(fv(xloc))
        return u1inf,v1inf

    def u_farm(self,abl,grid,u1r,v1r):
        '''
        Calculate perturbation velocity for wake model based on farm-averaged
        coupling strategy, i.e., based on farm-averaged velocity

        Parameters
        ----------
        abl: ABL object (defined in TLM.py)
            atmospheric state
        grid: Grid object (defined in TLM.py)
            numerical grid
        u1r, v1r: 1d numpy array
            velocity perturbation fields (in real space)

        Returns
        -------
        u1inf,v1inf: float
            perturbation velocity in x and y for wake model
        '''
        fu = interpolate.interp1d(grid.xs,u1r)
        fv = interpolate.interp1d(grid.xs,v1r)
        xs = np.linspace(self.xstart,self.xend,int(self.length/grid.dx))
        u1inf = np.mean(fu(xs))
        v1inf = np.mean(fv(xs))
        return u1inf,v1inf

    def firstTurbine(self):
        '''
        Find first turbine

        Returns
        -------
        _: integer
            index of the first turbine
        '''
        #Find first turbine (along x direction)
        x = np.array([self.turbines[i].x for i in range(self.Nturb)])
        return np.argmin(x)

    @property
    def fringe(self):
        '''fringe region forcing model'''
        return self.__fringe

class turbine(object):
    '''
    Turbine object (only for 2D grids)
    '''
    def __init__(self,xloc,yloc,diameter,thrustcoefficient,zhs):
        '''
        Parameters
        ----------
        xloc,yloc: float
            x and y coordinate
        diameter: float
            rotor diameter
        thrustcoefficient: float
            turbine thrust coefficient Ct
        zhs: float
            turbine hub height
        '''
        self.__x = xloc
        self.__y = yloc
        self.__D = diameter
        self.__Ct = thrustcoefficient
        self.__zhs = zhs

    @property
    def x(self):
        '''wind turbine x-coordinate'''
        return self.__x
    @property
    def y(self):
        '''wind turbine y-coordinate'''
        return self.__y
    @property
    def zhs(self):
        '''wind turbine hub heigth'''
        return self.__zhs
    @property
    def D(self):
        '''rotor diameter'''
        return self.__D
    @property
    def rotorarea(self):
        '''rotor swept area'''
        return np.pi/4.0*self.D**2
    @property
    def Ct(self):
        '''turbine thrust coefficient'''
        return self.__Ct
    @property
    def induction(self):
        '''axial induction factor (according to axial momentum theory)'''
        return 0.5-0.5*np.sqrt(1-self.Ct)
    @property
    def Cp(self):
        '''power coefficient (according to axial momentum theory)'''
        return 4*self.induction*(1-self.induction)**2

class turbine1D(turbine):
    '''
    Special 1D turbine object

    Geometrical footprint is a line at y=0
    '''
    def footprint(self,grid,L):
        '''
        Geometrical footprint of the turbine thrust force for the given grid,
        computed with a Gaussian filter

        Parameters
        ----------
        grid: Grid object (defined in TLM.py)
            Numerical grid
        L: float
            Gaussian filter length

        Returns
        -------
        R: numpy array with same shape as grid
            Geometrical footprint of the perturbing force
        '''
        dist = (grid.xs-self.x)**2+(self.y)**2
        R = 1./(np.pi*L**2)*np.exp(-dist/L**2)
        return R
