"""
Wind farm class for APM
"""

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "November 7, 2022"


import numpy as np

from wayve.forcing.apm_forcing import ForcingTerm
import wayve.forcing.wind_farms.wf_tools
from wayve.forcing.wind_farms.wf_tools import wind_farm_shape, footprint


class WindFarm(ForcingTerm):
    '''
    Wind farm object

    Wind-farm perturbing force model with individual turbines and Gaussian filtering
    '''

    def __init__(self, turbines, Lfilter, coupling, fudge_factor=1.):
        '''
        Initialise the wind farm model with choice of wake model and coupling

        Parameters
        ----------
        turbines: list of Turbine objects
            turbines of the wind farm
        Lfilter: float
            filter length for the Gaussian filter
        coupling: class inheriting from "Coupling"
            coupling method
        fudge_factor: float
            fudge factor
        '''
        # List of turbines
        self.__Nturb = len(turbines)
        self.__turbines = turbines
        # Wake model coupling
        self.__coupling = coupling
        # Convex polygon around the farm
        vertices, area = wind_farm_shape(self)
        self.__vertices = vertices
        self.__area = area
        # Other variables
        self.__Lfilter = Lfilter
        self.__fudge_factor = fudge_factor
        # Variables that will be computed when the object is pre-processed
        self.__footprint = None

    def preprocess(self, model):
        '''
        Perform a calculation of the wake model, assuming background flow.
        Initialise the Turbine footprint. The footprint is a list of sparse matrix.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        '''
        # Pre-process wake model and coupling method
        self.coupling.preprocess(model, self)
        # Initialize turbine footprints
        grid_deal = model.grid.deal_grid()
        self.__footprint = [0] * self.Nturb
        for index, turb in enumerate(self.turbines):
            self.__footprint[index] = footprint(grid_deal.xs, grid_deal.ys, grid_deal.dx, grid_deal.dy, self.Lfilter,
                                                self.xs[index], self.ys[index])

    def reprocess(self, model, result):
        '''
        Re-calculate the wake model, given the APM state.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        result: dict
            State of the APM variables
        '''
        # Re-process wake model and coupling method
        self.coupling.reprocess(model, self, result)

    def ZOC(self, model):
        """
        Compute the zero-order forcing terms of the APM.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zero-order perturbation terms.
        """
        H1 = model.abl.H1
        Nx = model.grid.Nx2
        Ny = model.grid.Ny

        # Compute 0th order forcing term (2D real)
        F0u, F0v = self.Force(model.grid.deal_grid())
        # Convert to fourier space, cast into 1D array and
        # divide by H1 (APM solves height-averaged equations)
        Bu = np.ravel(model.grid.r2c_deal(F0u))/H1
        Bv = np.ravel(model.grid.r2c_deal(F0v))/H1

        return np.concatenate((Bu,Bv,np.zeros((4*Nx*Ny,),dtype=np.complex128)))

    def FOC(self, model, x):
        """
        Compute the first-order contributions of this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Perturbation array, containing u1, v1, u2, v2, eta1, and eta2.
        Returns
        -------
        A 6*N2 array containing the first-order contribution terms in Fourier space.
        """
        N = model.grid.N2
        F = 0. * x
        # Change in F #
        # Extract dependent variables (u1,v1) from the vector
        u1c = x[0:N]
        v1c = x[N:2*N]
        # Convert to real space
        u1r = model.grid.c2r_deal(u1c.reshape(model.grid.shape2))
        v1r = model.grid.c2r_deal(v1c.reshape(model.grid.shape2))
        F1u, F1v = self.F1(model.abl, model.grid.deal_grid(), u1r, v1r)
        # Height-average
        F1u /= model.abl.H1
        F1v /= model.abl.H1
        # Output
        F[0:N] = np.ravel(model.grid.r2c_deal(F1u))
        F[N:2*N] = np.ravel(model.grid.r2c_deal(F1v))
        # Change in H #
        # Extract dependent variable eta1 from the vector
        eta1c = x[4*N:5*N]
        # Convert to real space
        eta1r = model.grid.c2r_deal(eta1c.reshape(model.grid.shape2))
        # Compute force
        F0u, F0v = self.Force(model.grid.deal_grid())
        # Derivative w.r.t. layer height
        dF0u = - F0u / model.abl.H1**2
        dF0v = - F0v / model.abl.H1**2
        # Compute in real space
        F1u_eta = dF0u * eta1r
        F1v_eta = dF0v * eta1r
        # Output
        F[0:N] += np.ravel(model.grid.r2c_deal(F1u_eta))
        F[N:2*N] += np.ravel(model.grid.r2c_deal(F1v_eta))
        return F

    def RHS(self, model, x):
        """
        Compute the full non-linear right-hand side of this forcing term.

        This method assumes the underlying wake model has already been processed to correspond to the given state x.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Perturbation array, containing u1, v1, u2, v2, eta1, and eta2.

        Returns
        -------
        A 6*N2 array containing the full right-hand side in Fourier space.
        """
        N = model.grid.N2
        # Compute 0th order forcing term (2D real)
        F0u, F0v = self.Force(model.grid.deal_grid())
        F0u = np.divide(F0u, model.abl.H1)
        F0v = np.divide(F0v, model.abl.H1)
        # Height-average #
        # Extract dependent variable eta1 from the vector
        eta1c = x[4*N:5*N]
        # Convert to real space
        eta1r = model.grid.c2r_deal(eta1c.reshape(model.grid.shape2))
        F0u += - np.divide(F0u, model.abl.H1) * eta1r
        F0v += - np.divide(F0v, model.abl.H1) * eta1r
        # Convert to fourier space, cast into 1D array and
        # divide by H1 (APM solves height-averaged equations)
        Bu = np.ravel(model.grid.r2c_deal(F0u))
        Bv = np.ravel(model.grid.r2c_deal(F0v))
        return np.concatenate((Bu, Bv, np.zeros((4*N,), dtype=np.complex128)))

    def Force(self, grid32):
        '''
        Compute  wind-farm force on specified grid

        Parameters
        ----------
        grid32: Grid object (defined in grid.py)
            numerical grid (de-aliased)

        Returns
        -------
        F0u, F0v: 2d numpy array
            Wind-farm force in x and y (same shape as grid)
        '''
        # Grid information
        Nx = grid32.Nx
        Ny = grid32.Ny
        # Output array
        F0 = np.zeros((2, Nx, Ny))
        # Turbine information
        rotorarea = [self.turbines[i].rotorarea for i in range(self.Nturb)]
        # Wake model output
        Ct = self.coupling.Ct
        St = self.coupling.St
        e_str = self.coupling.et
        # Calculate wind farm force
        for index in range(self.Nturb):
            F0 = wayve.forcing.wind_farms.wf_tools.evaluate_F(F0[0], F0[1],
                                                              Ct[index], rotorarea[index],
                                                              St[index], e_str[index],
                                                              self.footprint[index][0],
                                                              self.footprint[index][1],
                                                              self.footprint[index][2])
        # Apply fudge factor
        F0u = F0[0] * self.fudge_factor
        F0v = F0[1] * self.fudge_factor
        return F0u, F0v

    def power_turbines(self, rho=1.225):
        '''
        The power output of all the turbines. Assumes the wake model has been pre- or reprocessed.

        Parameters
        ----------
        rho (optional): float
            Air density (default: 1.225)
            Since the air density is not used in running the APM, it has to be provided by the user here.

        Returns
        -------
        _: array-like
            Power output of the turbines
        '''
        Nt = self.Nturb
        power = np.empty(Nt)
        St = self.coupling.St
        for t in range(Nt):
            Cp = self.turbines[t].Cp(St[t])
            rotorarea = self.turbines[t].rotorarea
            power[t] = rho * 0.5 * Cp * rotorarea * St[t] ** 3
        return power

    @property
    def xs(self):
        '''Wind turbine x-coordinates'''
        return np.array([turbine.x for turbine in self.turbines])

    @property
    def ys(self):
        '''Wind turbine y-coordinates'''
        return np.array([turbine.y for turbine in self.turbines])

    @property
    def Lfilter(self):
        '''Gaussian filter length'''
        return self.__Lfilter

    @Lfilter.setter
    def Lfilter(self, value):
        self.__Lfilter = value

    @property
    def coupling(self):
        '''Coupling strategy'''
        return self.__coupling

    @coupling.setter
    def coupling(self, value):
        self.__coupling = value

    @property
    def Nturb(self):
        '''Number of turbines'''
        return len(self.turbines)

    @property
    def turbines(self):
        '''List of Turbine objects'''
        return self.__turbines

    @property
    def footprint(self):
        '''Wind Turbine footprint, as defined on the 3/2 grid'''
        return self.__footprint

    @property
    def length(self):
        '''Length of area covered by the turbines'''
        return self.xend - self.xstart

    @property
    def width(self):
        '''Width of area covered by the turbines'''
        return self.yend - self.ystart

    @property
    def xstart(self):
        '''x coordinate of the start of the wind-farm area'''
        return self.xs.min()

    @property
    def xend(self):
        '''x coordinate of the end of the wind-farm area'''
        return self.xs.max()

    @property
    def ystart(self):
        '''y coordinate of the start of the wind-farm area'''
        return self.ys.min()

    @property
    def yend(self):
        '''y coordinate of the end of the wind-farm area'''
        return self.ys.max()

    @property
    def xcentre(self):
        '''x coordinate of the centre of the wind-farm area'''
        return (self.xstart + self.xend) / 2.0

    @property
    def ycentre(self):
        '''y coordinate of the centre of the wind-farm area'''
        return (self.ystart + self.yend) / 2.0

    @property
    def vertices(self):
        '''vertices of a convex polygon enclosing the wind farm'''
        return self.__vertices

    @property
    def area(self):
        '''area of a convex polygon enclosing the wind farm'''
        return self.__area

    @property
    def fudge_factor(self):
        """Fudge factor modifying the forcing"""
        return self.__fudge_factor

    @fudge_factor.setter
    def fudge_factor(self, value):
        self.__fudge_factor = value


class Turbine(object):
    '''
    Turbine object (only for 2D grids)
    '''

    def __init__(self, xloc, yloc, diameter, zh, ct, Cp_curve=None):
        '''
        Parameters
        ----------
        xloc,yloc: float
            x and y coordinate
        diameter: float
            rotor diameter
        zh: float
            turbine hub height
        ct: callable or float
            turbine thrust coefficient curve, as a function of inflow velocity
        Cp_curve: callable
            turbine power coefficient curve, as a function of inflow velocity (default=None)
            will be set to None if ct is a constant value
        '''
        self.__x = xloc
        self.__y = yloc
        self.__D = diameter
        self.__zh = zh
        if callable(ct):
            self.__Ct_curve = ct
        else:
            self.set_constant_Ct(ct)
        self.__Cp_curve = Cp_curve

    @property
    def x(self):
        '''wind turbine x-coordinate'''
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property
    def y(self):
        '''wind turbine y-coordinate'''
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value

    @property
    def zh(self):
        '''wind turbine hub height'''
        return self.__zh

    @zh.setter
    def zh(self, value):
        self.__zh = value

    @property
    def D(self):
        '''rotor diameter'''
        return self.__D

    @D.setter
    def D(self, value):
        self.__D = value

    @property
    def rotorarea(self):
        '''rotor swept area'''
        return np.pi / 4.0 * self.D ** 2

    def Ct(self, u):
        '''turbine Ct curve'''
        return self.__Ct_curve(u)

    def Cp(self, u):
        '''power coefficient (according to axial momentum theory)'''
        if self.__Cp_curve is not None:
            return self.__Cp_curve(u)
        else:
            Ct = self.Ct(u)
            ind = Turbine.induction(Ct)
            return 4 * ind * (1 - ind) ** 2

    def Ct_prime(self, u):
        '''disk-based thrust coefficient'''
        Ct = self.Ct(u)
        return Ct / (1 - Turbine.induction(Ct)) ** 2

    @staticmethod
    def induction(ct):
        '''axial induction factor (according to axial momentum theory)'''
        return 0.5 - 0.5 * np.sqrt(1 - ct)

    def set_constant_Ct(self, ct):
        '''Set the turbine thrust coefficient to the given constant value'''
        # Brief check on input value
        if not (0 <= ct and ct <= 1):
            raise ValueError('C_T should be between 0 and 1.')
        # Set up curve
        ct_curve = lambda u: 0 * u + ct     # Formulation compatible with floats and numpy arrays
        # Change object state
        self.__Ct_curve = ct_curve
        self.__Cp_curve = None
        return
