#!/usr/bin/env python

"""
Forcing models for APM
"""

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "August 7, 2017"

import numpy as np

from wayve.forcing.forcing_tools import fill_shape


class ForcingTerm(object):
    """
    Common class that sets up the interface for all APM forcing terms, or composites of forcing terms. This allows the
    rest of the APM to treat all types of forcing terms, or the combination of multiple forcing terms, as a single
    object through a uniform interface.
    """

    def preprocess(self, model):
        """
        Standard function for preprocessing

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        """
        pass

    def reprocess(self, model, result):
        """
        Standard function for reprocessing

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        result: dict
            State of the APM variables
        """
        pass

    def ZOC(self, model):
        """
        Compute the zeroth-order contributions of this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zero-order perturbation terms in Fourier space.
        """
        N = model.grid.N2
        return np.zeros(6*N, dtype=np.complex128)

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
        return 0. * x

    def RHS(self, model, x):
        """
        Compute the full right-hand side of this forcing term. This defaults to the sum of the zeroth- and first-order
        contributions, but subclasses can include non-linear terms.

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
        return self.ZOC(model) + self.FOC(model, x)

    def power(self, model, x):
        """
        Compute the power extracted from the ABL by this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Perturbation array, containing u1, v1, u2, v2, eta1, and eta2.

        Returns
        -------
        p   float
                The power extracted from the ABL.
        """
        # ABL object
        abl = model.abl
        # Grid object
        grid = model.grid
        N = grid.N2
        # Extract dependent variables from the state vector
        u1c = x[0:N].reshape(grid.shape2)
        v1c = x[1*N:2*N].reshape(grid.shape2)
        u2c = x[2*N:3*N].reshape(grid.shape2)
        v2c = x[3*N:4*N].reshape(grid.shape2)
        eta1c = x[4*N:5*N].reshape(grid.shape2)
        eta2c = x[5*N:].reshape(grid.shape2)
        # Get full variables in real space, using de-aliasing to avoid convolution issues
        u1r = abl.U1 + grid.c2r_deal(u1c)
        v1r = abl.V1 + grid.c2r_deal(v1c)
        u2r = abl.U2 + grid.c2r_deal(u2c)
        v2r = abl.V2 + grid.c2r_deal(v2c)
        h1r = abl.H1 + grid.c2r_deal(eta1c)
        h2r = abl.H2 + grid.c2r_deal(eta2c)
        # Compute RHS, in Fourier space
        rhs = self.RHS(model, x)
        rhs_u1 = rhs[0:N].reshape(grid.shape2)
        rhs_v1 = rhs[1*N:2*N].reshape(grid.shape2)
        rhs_u2 = rhs[2*N:3*N].reshape(grid.shape2)
        rhs_v2 = rhs[3*N:4*N].reshape(grid.shape2)
        # Convert to real space, using de-aliasing to avoid convolution issues
        rhs_u1r = grid.c2r_deal(rhs_u1)
        rhs_v1r = grid.c2r_deal(rhs_v1)
        rhs_u2r = grid.c2r_deal(rhs_u2)
        rhs_v2r = grid.c2r_deal(rhs_v2)
        # Compute different power contributions
        pow_u1r = np.multiply(np.multiply(u1r, rhs_u1r), h1r)
        pow_v1r = np.multiply(np.multiply(v1r, rhs_v1r), h1r)
        pow_u2r = np.multiply(np.multiply(u2r, rhs_u2r), h2r)
        pow_v2r = np.multiply(np.multiply(v2r, rhs_v2r), h2r)
        # De-aliasing grid
        grid32 = grid.deal_grid()
        # Integrate over the domain
        pow_u1 = np.trapz(np.trapz(pow_u1r, dx=grid32.dx, axis=0), dx=grid32.dy)
        pow_v1 = np.trapz(np.trapz(pow_v1r, dx=grid32.dx, axis=0), dx=grid32.dy)
        pow_u2 = np.trapz(np.trapz(pow_u2r, dx=grid32.dx, axis=0), dx=grid32.dy)
        pow_v2 = np.trapz(np.trapz(pow_v2r, dx=grid32.dx, axis=0), dx=grid32.dy)
        # Sum of contributions
        p = pow_u1 + pow_v1 + pow_u2 + pow_v2
        # Multiply with density
        p *= abl.rho
        return p

    def pressure_contribution(self, model, x):
        """
        Compute the pressure contribution of this forcing object in Fourier space.

        By default, this contribution is zero. However, cases where the pressure contribution of a forcing object is not
        zero could be:
            - Hills
            - Wind farms (through hydrodynamic effects of subgrid velocities, not yet implemented)

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Solution vector of the APM

        Returns
        -------
        Two N2 arrays containing the pressure contributions in Fourier space.
        """
        shape = model.grid.shape2
        return np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128)

    def eta_c(self, grid):
        """
        Return the Fourier transform of the topography of this forcing term over a given grid.
        This is only non-zero for Topography objects.
        """
        shape = grid.shape2
        return np.zeros(shape, dtype=np.complex128)


class ForcingComposite(ForcingTerm):
    """
    Class for composites of APM forcing terms. The composites can follow a nested structure.

    There are not many safety checks on what is added to the composite. It is up to the user to ensure that a
    ForcingComposite does not appear as one of its own children.
    """

    def __init__(self, children=None):
        if children is None:
            children = []
        self.__children = []
        for child in children:
            self.__children.append(child)

    def preprocess(self, model):
        """
        Standard function for preprocessing

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        """
        for term in self.__children:
            term.preprocess(model)
        return

    def reprocess(self, model, result):
        """
        Standard function for reprocessing

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        result: dict
            State of the APM variables
        """
        for term in self.__children:
            term.reprocess(model, result)
        return

    def ZOC(self, model):
        """
        Compute the zero-order forcing terms of the APM.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        Returns
        -------
        A 6*N2 array containing the zero-order perturbation terms in Fourier space.
        """
        N = model.grid.N2
        F = np.zeros(6*N, dtype=np.complex128)
        for term in self.children:
            F += term.ZOC(model)
        return F

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
        F = 0. * x
        for term in self.children:
            F += term.FOC(model, x)
        return F

    def RHS(self, model, x):
        """
        Compute the full right-hand side of this forcing term.

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
        F = 0. * x
        for term in self.children:
            F += term.RHS(model, x)
        return F

    def power(self, model, x):
        """
        Compute the power extracted from the ABL by this forcing term.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Perturbation array, containing u1, v1, u2, v2, eta1, and eta2.

        Returns
        -------
        p   float
                The power extracted from the ABL.
        """
        p = 0.
        for term in self.children:
            p += term.power(model, x)
        return p

    def pressure_contribution(self, model, x):
        """
        Compute the pressure contribution of this forcing object in Fourier space.

        By default, this contribution is zero. However, cases where the pressure contribution of a forcing object is not
        zero could be:
            - Hills
            - Wind farms (through hydrodynamic effects of subgrid velocities, not yet implemented)

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        x       Numpy array
                    Solution vector of the APM

        Returns
        -------
        Two N2 arrays containing the pressure contributions in Fourier space.
        """
        shape = model.grid.shape2
        p1 = np.zeros(shape, dtype=np.complex128)
        p2 = np.zeros(shape, dtype=np.complex128)
        for term in self.children:
            p1_term, p2_term = term.pressure_contribution(model, x)
            p1 += p1_term
            p2 += p2_term
        return p1, p2

    def eta_c(self, grid):
        """
        Return the Fourier transform of the topography of this forcing term over a given grid.
        This is only non-zero for Topography objects.
        """
        shape = grid.shape2
        field = np.zeros(shape, dtype=np.complex128)
        for term in self.children:
            field += term.eta_c(grid)
        return field

    @property
    def children(self):
        """
        Return a list containing the forcing terms of this composite.
        """
        return self.__children.copy()

    def add_children(self, children):
        """
        Add new forcing terms to this composite. Be careful with the input! Make sure the ForcingComposite doesn't
        contain itself or doubles after this operation, the checks are not fool-proof.

        Parameters
        ----------
        child   List of ForcingTerms
                    New forcing terms to be added to this composite
        """
        if children is None:
            children = []
        for child in children:
            self.add_child(child)

    def add_child(self, child):
        """
        Add a new forcing term to this composite. Be careful with the input! Make sure the ForcingComposite doesn't
        contain itself or doubles after this operation, the checks are not fool-proof.

        Parameters
        ----------
        child   ForcingTerm
                    New forcing to be added to this composite
        """
        if not isinstance(child, ForcingTerm):
            raise ValueError("New forcing terms added to ForcingComposite must be subclasses of ForcingTerm.")
        if child in self.children:
            raise ValueError("New forcing term is already in this ForcingComposite.")
        if child is self:
            raise ValueError("A ForcingComposite can not be added to itself.")
        self.__children.append(child)


class CST(ForcingTerm):
    '''
    Common class for perturbing force models with a constant drag coefficient
    '''

    def __init__(self, CT, vertices):
        '''
        Initialise a constant drag coefficient forcing with the given shape.

        For now, only convex polygon shapes are supported. The points describing the vertices of the polygon should be
        in counter-clockwise order, and should form a closed loop, so that the first and last points are the same.

        Parameters
        ----------
        CT: float
            thrust coefficient of the forcing region
        vertices: np array
            Vertices of the polygon as a closed loop, ordered counter-clockwise, shape (npoints,2)
        '''
        # Forcing parameters
        self.__CT = CT
        self.__vertices = vertices

    def preprocess(self, model):
        '''
        Standard function for preprocessing

        Constant drag coefficient models do not require preprocessing,
        so this function is empty

        Parameters
        ----------
        model
        '''
        pass

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
        # ABL object
        abl = model.abl
        # Grid object
        grid = model.grid
        N = grid.N2
        # Compute 0th order forcing term (2D real space)
        F0x, F0y = self.F0(abl, grid)
        # Divide by H1 (APM solves height-averaged equations)
        F0x /= abl.H1
        F0y /= abl.H1
        # Convert to fourier space and cast into 1D array
        Bx = np.ravel(grid.r2c(F0x))
        By = np.ravel(grid.r2c(F0y))
        return np.concatenate((Bx, By, np.zeros((4*N,), dtype=np.complex128)))

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
        # ABL object
        abl = model.abl
        # Grid object
        grid = model.grid
        N = grid.N2
        # Extract dependent variables (u1,v1) from the vector
        u1c = x[0:N].reshape(grid.shape2)
        v1c = x[N:2*N].reshape(grid.shape2)
        # De-aliasing grid
        grid32 = grid.deal_grid()
        # Convert to real space, using de-aliasing to avoid convolution issues
        u1r = grid.c2r_deal(u1c)
        v1r = grid.c2r_deal(v1c)
        # Compute first-order force, in real space
        F1x, F1y = self.F1(abl, grid32, u1r, v1r)
        # Height-average
        F1x /= abl.H1
        F1y /= abl.H1
        # Output
        B = 0. * x
        B[0:N] = np.ravel(grid.r2c_deal(F1x))
        B[N:2*N] = np.ravel(grid.r2c_deal(F1y))
        # Change in H #
        # Extract dependent variable eta1 from the vector
        eta1c = x[4*N:5*N].reshape(grid.shape2)
        # Convert to real space
        eta1r = grid.c2r_deal(eta1c)
        # Compute force
        F0u, F0v = self.F0(abl, grid32)
        # Derivative w.r.t. layer height
        dF0u = - F0u / abl.H1**2
        dF0v = - F0v / abl.H1**2
        # Compute in real space
        F1u_eta = dF0u * eta1r
        F1v_eta = dF0v * eta1r
        # Output
        B[0:N] += np.ravel(grid.r2c_deal(F1u_eta))
        B[N:2*N] += np.ravel(grid.r2c_deal(F1v_eta))
        return B

    def RHS(self, model, x):
        """
        Compute the full non-linear right-hand side of this forcing term.

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
        # ABL object
        abl = model.abl
        # Grid object
        grid = model.grid
        N = grid.N2
        # Extract dependent variables (u1,v1,eta1) from the vector
        u1c = x[0:N].reshape(grid.shape2)
        v1c = x[N:2*N].reshape(grid.shape2)
        eta1c = x[4*N:5*N].reshape(grid.shape2)
        # De-aliasing grid
        grid32 = grid.deal_grid()
        # Convert to real space, using de-aliasing to avoid convolution issues
        u1r = grid.c2r_deal(u1c)
        v1r = grid.c2r_deal(v1c)
        eta1r = grid.c2r_deal(eta1c)
        # Compute total force, in real space
        Ftx, Fty = self.Ftot(abl, grid32, u1r, v1r)
        # Divide by h1 (APM solves height-averaged equations)
        h1 = abl.H1 + eta1r
        Ftx = np.divide(Ftx, h1)
        Fty = np.divide(Fty, h1)
        # Convert to fourier space and cast into 1D array
        Bx = np.ravel(model.grid.r2c_deal(Ftx))
        By = np.ravel(model.grid.r2c_deal(Fty))
        return np.concatenate((Bx, By, np.zeros((4*N,), dtype=np.complex128)))

    def F0(self, abl, grid):
        '''
        Compute the zero-th order term of the drag force on the given numerical grid

        Parameters
        ----------
        abl: ABL object
            Atmospheric state
        grid: Grid object
            Numerical grid

        Returns
        -------
        F0x, F0y: numpy array with same shape as grid
            Zero-th order term of the drag force (real space)
        '''
        F0x = self.CT * abl.S1 * abl.U1 * self.footprint(grid)
        F0y = self.CT * abl.S1 * abl.V1 * self.footprint(grid)
        return F0x, F0y

    def F1(self, abl, grid, u1r, v1r):
        '''
        Compute the first order term of the drag force on the given numerical grid

        Parameters
        ----------
        abl: ABL object
            Atmospheric state
        grid: Grid object
            Numerical grid
        u1r,v1r: numpy array with same shape as grid
            Perturbation velocities

        Returns
        -------
        F1x, F1y: numpy array with same shape as grid
            First order term of the drag force (real space)
        '''
        F1x = self.CT * (abl.S1 + abl.U1 ** 2 / abl.S1) * u1r * self.footprint(grid)
        F1x += self.CT * abl.U1 * abl.V1 / abl.S1 * v1r * self.footprint(grid)
        F1y = self.CT * (abl.S1 + abl.V1 ** 2 / abl.S1) * v1r * self.footprint(grid)
        F1y += self.CT * abl.U1 * abl.V1 / abl.S1 * u1r * self.footprint(grid)
        return F1x, F1y

    def Ftot(self, abl, grid, u1r, v1r):
        '''
        Compute the total drag force on the given numerical grid

        Parameters
        ----------
        abl: ABL object
            Atmospheric state
        grid: Grid object
            Numerical grid
        u1r,v1r: numpy array with same shape as grid
            Perturbation velocities

        Returns
        -------
        Ftx, Fty: numpy array with same shape as grid
            First order term of the drag force (real space)
        '''
        # Wind speeds
        u = abl.U1 + u1r
        v = abl.V1 + v1r
        s = np.sqrt(np.square(u) + np.square(v))
        # Forces
        Ftx = self.CT * s * u * self.footprint(grid)
        Fty = self.CT * s * v * self.footprint(grid)
        return Ftx, Fty

    def footprint(self, grid):
        '''
        Compute the geometrical footprint of the perturbing force for the given grid

        Parameters
        ----------
        grid: Grid object (defined in apm.py)
            Numerical grid

        Returns
        -------
        footprint: numpy array with same shape as grid
            Geometrical footprint of the perturbing force
        '''
        # Grid
        xm, ym = np.meshgrid(grid.xs, grid.ys, indexing='ij')
        # Footprint calculation
        footprint = fill_shape(self.vertices, xm, ym)
        return footprint

    @property
    def CT(self):
        '''Drag coefficient'''
        return self.__CT

    @property
    def vertices(self):
        '''Vertices of a convex polygon defining the shape of the forcing'''
        return self.__vertices
