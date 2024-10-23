"""
File containing different ways of coupling wake models with non-uniform background flows to the APM.
"""

__author__ = "Koen Devesse"
__date__ = "July 14, 2022"

import numpy as np
from scipy import interpolate
from scipy.linalg import norm
from scipy.interpolate import RegularGridInterpolator
from numba import njit

from wayve.forcing.wind_farms.wake_model_coupling.coupling import Coupling
from wayve.forcing.wind_farms.wf_tools import height_average, filter_2d, filter_3d, filter_2d_numba


class VaryingBackground(Coupling):
    """
    A Coupling that couples the APM to wake models with a varying background velocity.

    These coupling objects follow the same structure as the superclass Coupling. After an initial calculation, through
    the method "preprocess", the results are stored, so they can be called on by the APM. These results can be updated
    based on a new APM state, using the method "reprocess".

    In addition to the interface provided by the base class, this class and its subclasses provide callables to evaluate
    the wake model background velocities and the APM state at requested locations in a region around the wind farm.

    Finally, this class also provides a mechanism for evaluating the wake model velocity on a dense grid around the wind
    farm. This is done through a WakeModelVelocityHandler object, which this class can have as an attribute. If this is
    the case, the Coupling object will call the Handler object to perform its calculations when necessary. If no Handler
    object is provided, these expensive calculations are avoided. This setup keeps the code related to these
    calculations out of the Coupling classes, simplifying their use and implementation.
    """

    def __init__(self, wake_model, wm_velocity_handler=None):
        """
        Initialize a VaryingBackground Coupling object.

        Parameters
        ----------
        wake_model  WakeModelInterface object (defined in wake_model_interface.py)
            Wake model that this Coupling interfaces to.
        wm_velocity_handler     WakeModelVelocityHandler object (optional)
            Handler for calculating and storing the wake model velocity fields (default: None)
        """
        # Basic coupling object setup
        super().__init__()
        self.__wake_model = wake_model
        # Wake model velocity evaluator
        self.__wm_velocity_handler = wm_velocity_handler
        # Background velocity functions
        self.__ub_evaluator = None
        self.__vb_evaluator = None
        # APM callables
        self.__apm_evaluator = None
        self.__h1_evaluator = None

    @property
    def wake_model(self):
        """
        The wake model object this coupling interfaces to.
        """
        return self.__wake_model

    @property
    def ub_evaluator(self):
        """
        Callable to evaluate the velocity scale of the background variations in the x-direction
        """
        return self.__ub_evaluator

    @ub_evaluator.setter
    def ub_evaluator(self, value):
        self.__ub_evaluator = value

    @property
    def vb_evaluator(self):
        """
        Callable to evaluate the velocity scale of the background variations in the x-direction
        """
        return self.__vb_evaluator

    @vb_evaluator.setter
    def vb_evaluator(self, value):
        self.__vb_evaluator = value

    @property
    def apm_evaluator(self):
        """
        Callable to evaluate the layer wind speeds
        """
        return self.__apm_evaluator

    @property
    def h1_evaluator(self):
        """
        Callable to evaluate the layer height
        """
        return self.__h1_evaluator

    @property
    def wm_velocity_handler(self):
        return self.__wm_velocity_handler

    def preprocess(self, model, wind_farm):
        """
        Initial calculation of the wake model, assuming background state.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        wind_farm   WindFarm object
                        Wind farm for which the calculation is performed
        """
        # APM components
        abl = model.abl
        # Set up layer height evaluator callable
        self.set_up_apm_evaluators(model, wind_farm)
        # Set up internal state for u_b, v_b
        self.initialize_ub_vb(model, wind_farm)
        # Set up background evaluator callable
        u_bg_evaluator = self.set_up_u_bg_evaluator(abl)
        # Perform wake model calculation
        St, Ct, et = self.wake_model.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, self.apm_evaluator)
        # Store wake model calculation results
        self.St = St
        self.Ct = Ct
        self.et = et
        # If used, run wake model velocity evaluation
        if self.wm_velocity_handler is not None:
            self.wm_velocity_handler.calculate_u_wm(wind_farm, abl, self.wake_model, u_bg_evaluator, self.apm_evaluator)

    def reprocess(self, model, wind_farm, result):
        """
        Update the wake model, given the new APM state.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        wind_farm   WindFarm object
                        Wind farm for which the calculation is performed
        result      dict
                        State of the APM variables, or equivalent. Must include:
                            - 'u1r': x-component of the lower layer velocity
                            - 'v1r': y-component of the lower layer velocity
                            - 'eta1r': lower layer interface displacement
                            - 'u2r': x-component of the upper layer velocity
                            - 'v2r': y-component of the upper layer velocity
                            - 'eta2r': upper layer thickness perturbation
                            - 'p1r': lower layer pressure perturbation
                            - 'p2r': upper layer pressure perturbation
        """
        # APM components
        abl = model.abl
        # Set up layer height evaluator callable
        self.set_up_apm_evaluators(model, wind_farm, result)
        # Calculate u_b, v_b based on the new APM state
        self.update_ub_vb(model, wind_farm, result)
        # Set up background evaluator callable
        u_bg_evaluator = self.set_up_u_bg_evaluator(abl)
        # Perform wake model calculation
        St, Ct, et = self.wake_model.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, self.apm_evaluator)
        # Store wake model calculation results
        self.St = St
        self.Ct = Ct
        self.et = et
        # If used, run wake model velocity evaluation
        if self.wm_velocity_handler is not None:
            self.wm_velocity_handler.calculate_u_wm(wind_farm, abl, self.wake_model, u_bg_evaluator, self.apm_evaluator)

    def set_up_u_bg_evaluator(self, abl):
        """
        Set up a callable that can calculate the background velocity components (u and v) at given locations.
        """

        # Define callable
        def u_bg_evaluator(locations):
            """
            Calculates the background velocity components (u and v) at given locations.

            Parameters
            ----------
            locations   numpy array
                A 2d numpy array specifying the locations where the background velocities have to be calculated.
                It has the shape (Nlocs,3), where Nlocs is the number of locations, so that locations[i,:] corresponds
                to the x, y, and z coordinates of location i.

            Returns
            -------
            velocities  numpy array
                A 2d array containing the background velocities.
                It has the shape (Nlocs,2), so that velocities[i,:] corresponds to the u and v velocity components at
                location i.
            """
            # Perform calculations
            u_b = self.ub_evaluator(locations[:, 0], locations[:, 1])
            v_b = self.vb_evaluator(locations[:, 0], locations[:, 1])
            f = self.vertical_profile(abl, locations[:, 2])
            u0_bg = abl.u(locations[:, 2])
            v0_bg = abl.v(locations[:, 2])
            # Set up output array
            N_loc = locations.shape[0]  # Assumed to have shape (N_loc, 3)
            velocities = np.empty((N_loc, 2))
            # Combine
            velocities[:, 0] = u0_bg + u_b * f
            velocities[:, 1] = v0_bg + v_b * f
            return velocities

        # Return callable
        return u_bg_evaluator

    def set_up_apm_evaluators(self, model, wind_farm, result=None):
        """
        Initialize the evaluator functions for u_1, v_1, and h_1 based on the given APM state.
        """
        # APM components
        abl = model.abl
        grid = model.grid
        # Callables for individual variables
        if result is None:
            # If no APM state given, use straightforward lambda functions
            f_u1 = lambda locs: 0. * locs[0] + model.abl.U1     # u_1 = U_1
            f_v1 = lambda locs: 0. * locs[0] + model.abl.V1     # v_1 = V_1
            f_h1 = lambda locs: 0. * locs[0] + model.abl.H1     # h_1 = H_1
        else:
            # APM states
            u1_full = abl.U1 + result["u1r"]
            v1_full = abl.V1 + result["v1r"]
            h1_full = abl.H1 + result["eta1r"]
            # Get region around wind farm
            x_sel, y_sel, selection = self.select_grid_around_farm(wind_farm, grid)
            # Select u_1, v_1, h_1 in relevant region
            shape_sel = (len(x_sel), len(y_sel))
            u1 = np.reshape(u1_full[selection], shape_sel)
            v1 = np.reshape(v1_full[selection], shape_sel)
            h1 = np.reshape(h1_full[selection], shape_sel)
            # Set up interpolation functions
            f_u1 = RegularGridInterpolator((x_sel, y_sel), u1, method="linear", bounds_error=False, fill_value=None)
            f_v1 = RegularGridInterpolator((x_sel, y_sel), v1, method="linear", bounds_error=False, fill_value=None)
            f_h1 = RegularGridInterpolator((x_sel, y_sel), h1, method="linear", bounds_error=False, fill_value=None)
        # Define final callable functions
        def apm_callable(x, y):
            """
            Evaluates the layer wind speeds and height at the requested locations

            Parameters
            ----------
            x, y    numpy arrays
                Coordinates at which the layer height is evaluated.
                x and y should have the same shape.

            Returns
            -------
            u1  numpy array
                A numpy array containing the layer wind speed in the x-direction at the requested locations.
                It has the same shape as x and y.
            v1  numpy array
                A numpy array containing the layer wind speed in the y-direction at the requested locations.
                It has the same shape as x and y.
            h1  numpy array
                A numpy array containing the layer height at the requested locations.
                It has the same shape as x and y.
            """
            return f_u1((x, y)), f_v1((x, y)), f_h1((x, y))
        def h1_callable(x, y):
            """
            Evaluates the layer height at the requested locations

            Parameters
            ----------
            x, y    numpy arrays
                Coordinates at which the layer height is evaluated.
                x and y should have the same shape.

            Returns
            -------
            h1  numpy array
                A numpy array containing the layer height at the requested locations.
                It has the same shape as x and y.
            """
            return f_h1((x, y))
        # Set callables as object properties
        self.__apm_evaluator = apm_callable
        self.__h1_evaluator = h1_callable

    def initialize_ub_vb(self, model, wind_farm):
        """
        Initialize the evaluator functions for u_b and v_b, so that u_b=v_b=0.
        """
        self.__ub_evaluator = lambda x, y: 0. * x
        self.__vb_evaluator = lambda x, y: 0. * x

    def update_ub_vb(self, model, wind_farm, result):
        """
        Update the evaluator functions for u_b and v_b, based on the given APM state.
        """
        raise Exception("Background velocity calculation not implemented!")

    def vertical_profile(self, abl, z):
        """
        Evaluate the vertical shape function used by this coupling.
        """
        # Parameters
        z0 = abl.z0
        kappa = 0.41
        # Logarithmic profile
        f = 1 / kappa * np.log(z / z0, out=np.zeros_like(z), where=(z >= z0))   # f=0 for z<=z0
        return f

    def region_around_farm(self, wind_farm):
        """
        Define the region region around the given wind farm over which calculation of the background velocities and
        layer height are guaranteed to be supported.
        """
        # Buffer region around wind farm
        edge_region = 2 * wind_farm.Lfilter
        # Region boundaries
        x_min = wind_farm.xstart - edge_region
        x_max = wind_farm.xend + edge_region
        y_min = wind_farm.ystart - edge_region
        y_max = wind_farm.yend + edge_region
        return x_min, x_max, y_min, y_max

    def select_grid_around_farm(self, wind_farm, grid):
        """
        Select a part of the APM grid in a region around the given wind farm.

        In order to avoid an expensive interpolations over the whole APM grid, this method implements the selection of a
        small portion around the wind farm.
        """
        # Region boundaries
        x_min, x_max, y_min, y_max = self.region_around_farm(wind_farm)
        # Select the APM gridpoints that form the smallest square around the domain
        x_min_apm = np.max(grid.xs[grid.xs < x_min])
        x_max_apm = np.min(grid.xs[grid.xs > x_max])
        y_min_apm = np.max(grid.ys[grid.ys < y_min])
        y_max_apm = np.min(grid.ys[grid.ys > y_max])
        # Region selection
        sel_x = np.logical_and(x_min_apm <= grid.xs, grid.xs <= x_max_apm)
        sel_y = np.logical_and(y_min_apm <= grid.ys, grid.ys <= y_max_apm)
        sel_x_mesh, sel_y_mesh = np.meshgrid(sel_x, sel_y, indexing='ij')
        selection = np.logical_and(sel_x_mesh, sel_y_mesh)
        # Grid selection
        x_sel = grid.xs[sel_x]
        y_sel = grid.ys[sel_y]
        return x_sel, y_sel, selection

    def reconstruct_apm(self, wind_farm, abl, grid, result):
        """
        Perform the inverse coupling calculation, i.e. compute the APM velocity field based on the background velocities
        calculated by this coupling object.

        This is mostly used for testing and computing residuals.

        This method assumes that this Coupling object has a wake model velocity evaluator object which has stored the
        correct turbine-level velocity fields.
        """
        if self.wm_velocity_handler is None:
            raise ValueError("No wake model velocity evaluator has been configured for this coupling!")
        # Get subgrid
        subgrid = self.wm_velocity_handler.subgrid
        # Get subgrid velocities
        u_sg = self.wm_velocity_handler.u_sg
        v_sg = self.wm_velocity_handler.v_sg
        # Get APM velocities
        u1_apm = abl.U1 + result["u1r"]
        v1_apm = abl.V1 + result["v1r"]
        # Get APM coordinates within subgrid domain
        x_c, y_c = subgrid.in_common(grid)
        # Perform APM operations on velocities within subgrid
        u_res = self.ha_and_filter(u_sg, subgrid, wind_farm.Lfilter, x_c, y_c, filter_first=True)
        v_res = self.ha_and_filter(v_sg, subgrid, wind_farm.Lfilter, x_c, y_c, filter_first=True)
        # Add filtered-in contributions
        u_res += filtered_in_contribution(grid, subgrid, u1_apm, wind_farm.Lfilter, x_c, y_c)
        v_res += filtered_in_contribution(grid, subgrid, v1_apm, wind_farm.Lfilter, x_c, y_c)
        return u_res, v_res

    def ha_and_filter(self, field_3d, subgrid, Lf, x_c, y_c, filter_first=False, zero_edge=True, use_scipy=False):
        """Height-average and filter the given 3D field.

        Parameters
        ----------
        field_3d: array-like
            3D field to be filtered
        subgrid: SubGrid object
            Fine 3D grid
        Lf: float
            Filter length
        x_c: array-like
            Coarse resolution x-coordinates
        y_c: array-like
            Coarse resolution y-coordinates
        filter_first: boolean (optional)
            Whether the filtering operation is applied before or after the height-averaging operator (default: False)
        zero_edge: boolean (optional)
            Whether the field is taken to be zero outside the domain bounds, or extrapolated (default: True)
        use_scipy: boolean (optional)
            Whether a scipy or numba implementation is used (default: False)
        """
        if not filter_first:    # First HA, then filter
            # Height-average
            ha = self.exact_height_average(field_3d, subgrid)
            # Filter
            ha_filt = filter_2d(ha, subgrid, Lf, x_c, y_c, zero_edge, use_scipy)
        else:   # First filter, then HA
            # Filter
            filt = filter_3d(field_3d, subgrid, Lf, x_c, y_c, zero_edge)
            # Get h1 on coarse grid
            x_m, y_m = np.meshgrid(x_c, y_c, indexing="ij")
            h1 = self.h1_evaluator(x_m, y_m)
            # Height-average
            ha_filt = height_average(filt, h1, subgrid.zs)
        return ha_filt

    def exact_height_average(self, field_3d, subgrid):
        """
        Return the height-average of the 3D field (x,y,z), up to the varying altitude abl.H1+eta.
        z should be ordered, and it's highest value should always be higher than max(abl.H1+eta)
        """
        # Read out subgrid
        x, y = subgrid.xy
        z = subgrid.zs
        # Get h1 on subgrid
        h1 = self.h1_evaluator(x, y)
        if np.any(h1 <= 0.):
            raise RuntimeError("Lower layer height reduced to zero!")
        if np.any(h1 >= z[-1]):
            raise RuntimeError("Lower layer height became too large!")
        # Height-average field
        field_ha = height_average(field_3d, h1, z)
        return field_ha


class PureWM(VaryingBackground):
    """
    A coupling that just uses the unperturbed background velocity, and leaves out APM effects.
    Mostly used to make analyses of the WM without APM feedback.
    """

    def update_ub_vb(self, model, wind_farm, result):
        """
        This coupling always takes u_b and v_b to be 0, so the default initialization performed in VaryingBackground
        does not have to be updated.
        """
        return

    def reconstruct_apm(self, wind_farm, abl, grid, result):
        """
        Perform the inverse coupling calculation, i.e. compute the APM velocity field based on the background velocities
        calculated by this coupling object.

        This method is overridden, so that the edge corrections using the APM velocity fields are not used.
        """
        if self.wm_velocity_handler is None:
            raise ValueError("No wake model velocity evaluator has been configured for this coupling!")
        # Get subgrid
        subgrid = self.wm_velocity_handler.subgrid
        # Get subgrid velocities
        u_sg = self.wm_velocity_handler.u_sg
        v_sg = self.wm_velocity_handler.v_sg
        # Get APM coordinates within subgrid domain
        x_c, y_c = subgrid.in_common(grid)
        # Perform APM operations on velocities within subgrid
        u_res = self.ha_and_filter(u_sg, subgrid, wind_farm.Lfilter, x_c, y_c, zero_edge=False, filter_first=True)
        v_res = self.ha_and_filter(v_sg, subgrid, wind_farm.Lfilter, x_c, y_c, zero_edge=False, filter_first=True)
        return u_res, v_res


@njit(parallel=False)
def height_average_shape_function(f, z, h):
    """
    Return the height-average of the shape function f from z=0 to z=h.

    Parameters
    ----------
    f   1d numpy array
            Shape function
    z   1d numpy array
            Vertical grid on which f is defined (same shape as f)
    h   2d numpy array
            Layer height

    Returns
    -------
    ..  2d numpy array
            The height-average of f (same shape as h)
    """
    # Set up output array
    Nx = h.shape[0]
    Ny = h.shape[1]
    f_ha = np.zeros(h.shape)
    # Loop over grid
    for i in range(Nx):
        for j in range(Ny):
            # Select grid within layer
            sel = z <= h[i, j]
            z_ij = z[sel]
            f_ij = f[sel]
            # Interpolate to layer boundary
            final_z = h[i, j]
            final_f = np.interp(h[i, j], z, f)     # Standard linear interpolation
            # Extend array to layer boundary
            z_ij = np.append(z_ij, final_z)
            f_ij = np.append(f_ij, final_f)
            # Height-average
            f_ha[i, j] = np.trapz(f_ij, x=z_ij) / (z_ij[-1] - z_ij[0])
    return f_ha


def filtered_in_contribution(grid, subgrid, field, L_f, x_c, y_c):
    # # # APM filtered-in term # # #
    # Domain boundaries #
    x_bound = np.array([subgrid.x_min, subgrid.x_max])
    y_bound = np.array([subgrid.y_min, subgrid.y_max])
    # Extract relevant sections of APM grid and field #
    # Region boundaries
    x_filt_min = x_bound[0] - 5. * L_f
    x_filt_max = x_bound[1] + 5. * L_f
    y_filt_min = y_bound[0] - 5. * L_f
    y_filt_max = y_bound[1] + 5. * L_f
    # Selection
    sel_x = np.logical_and(x_filt_min <= grid.xs, grid.xs <= x_filt_max)
    sel_y = np.logical_and(y_filt_min <= grid.ys, grid.ys <= y_filt_max)
    x_rel = grid.xs[sel_x]
    y_rel = grid.ys[sel_y]
    field_rel = field[sel_x, :]
    field_rel = field_rel[:, sel_y]
    # Add gridpoints at domain boundaries to get sharp edge of delta_WF #
    # Spacing of additional gridpoints
    dx_filt_in = L_f / 2000
    # Add gridpoints in x-direction
    x_edge = np.array([x_bound[0]-dx_filt_in, x_bound[0]+dx_filt_in,
                       x_bound[1]-dx_filt_in, x_bound[1]+dx_filt_in])
    x_s = np.append(x_rel, x_edge)
    x_s.sort()
    # Add gridpoints in y-direction
    y_edge = np.array([y_bound[0]-dx_filt_in, y_bound[0]+dx_filt_in,
                       y_bound[1]-dx_filt_in, y_bound[1]+dx_filt_in])
    y_s = np.append(y_rel, y_edge)
    y_s.sort()
    # Interpolate field onto grid with added points
    f = interpolate.interp2d(x_rel, y_rel, field_rel.T, kind="linear")
    field_s = f(x_s, y_s).T
    # Set field to zero inside domain #
    # Determine gridpoints in WindFarm
    out_dom = np.zeros(field_s.shape)
    out_dom[x_s <= x_bound[0], :] = 1.
    out_dom[x_s >= x_bound[1], :] = 1.
    out_dom[:, y_s <= y_bound[0]] = 1.
    out_dom[:, y_s >= y_bound[1]] = 1.
    # Set field to zero inside WindFarm
    field_s_out = np.multiply(field_s, out_dom)
    # Set up meshgrid
    x_m, y_m = np.meshgrid(x_s, y_s, indexing='ij')
    # Filter field
    filt_in_term = filter_2d_numba(field_s_out, x_c, y_c, x_s, y_s, x_m, y_m, L_f)
    return filt_in_term


class SubGrid:
    """
    Class of 3D grids, used to represent flow fields at higher resolutions than the APM grid.
    """

    def __init__(self, Nx, Ny, Nz, x_min, x_max, y_min, y_max, z_min, z_max):
        """
        Initialize a 3D grid object.

        Parameters
        ----------
        Nx      int
            Number of gridpoints in the x-direction
        Ny      int
            Number of gridpoints in the y-direction
        Nz      int
            Number of gridpoints in the z-direction
        x_min   float
            Lower edge of the grid in the x-direction
        x_max   float
            Upper edge of the grid in the x-direction
        y_min   float
            Lower edge of the grid in the y-direction
        y_max   float
            Upper edge of the grid in the y-direction
        z_min   float
            Lower edge of the grid in the z-direction
        z_max   float
            Upper edge of the grid in the z-direction
        """
        # Grid size
        self.__Nx = Nx
        self.__Ny = Ny
        self.__Nz = Nz
        # Grid boundaries
        self.__x_min = x_min
        self.__x_max = x_max
        self.__y_min = y_min
        self.__y_max = y_max
        self.__z_min = z_min
        self.__z_max = z_max

    def in_domain(self, field, grid):
        """Select the part of the given field defined on the given grid that horizontally overlaps with this subgrid"""
        sel_x = np.logical_and(self.x_min <= grid.xs, grid.xs <= self.x_max)
        sel_y = np.logical_and(self.y_min <= grid.ys, grid.ys <= self.y_max)
        field_id = field[sel_x, :]
        field_id = field_id[:, sel_y]
        return field_id

    def in_common(self, grid):
        """Select the part of the given grid that horizontally overlaps with this subgrid"""
        sel_x = np.logical_and(self.x_min <= grid.xs, grid.xs <= self.x_max)
        sel_y = np.logical_and(self.y_min <= grid.ys, grid.ys <= self.y_max)
        x_ic = grid.xs[sel_x]
        y_ic = grid.ys[sel_y]
        return x_ic, y_ic

    def interp_on_grid(self, field, x_c, y_c):
        """Interpolate the given field onto this subgrid"""
        # Set up interpolation function
        f = RegularGridInterpolator((x_c, y_c), field,
                                    method="linear", bounds_error=False, fill_value=None)
        # Set up interpolation grid
        x_g, y_g = np.meshgrid(self.xs, self.ys, indexing='ij')
        # Interpolate onto grid
        y = f((x_g, y_g))
        return y

    @property
    def Nx(self):
        """Number of gridpoints in the x-direction"""
        return self.__Nx

    @property
    def Ny(self):
        """Number of gridpoints in the y-direction"""
        return self.__Ny

    @property
    def Nz(self):
        """Number of gridpoints in the z-direction"""
        return self.__Nz

    @property
    def Nlocs(self):
        """Total number of gridpoints"""
        return self.Nx * self.Ny * self.Nz

    @property
    def x_min(self):
        """Lower edge of the grid in the x-direction"""
        return self.__x_min

    @property
    def x_max(self):
        """Upper edge of the grid in the x-direction"""
        return self.__x_max

    @property
    def y_min(self):
        """Lower edge of the grid in the y-direction"""
        return self.__y_min

    @property
    def y_max(self):
        """Upper edge of the grid in the y-direction"""
        return self.__y_max

    @property
    def z_min(self):
        """Lower edge of the grid in the z-direction"""
        return self.__z_min

    @property
    def z_max(self):
        """Upper edge of the grid in the z-direction"""
        return self.__z_max

    @property
    def Lx(self):
        """Grid length in the x-direction"""
        return self.x_max - self.x_min

    @property
    def Ly(self):
        """Grid length in the y-direction"""
        return self.y_max - self.y_min

    @property
    def xcenter(self):
        """Grid center in the x-direction"""
        return (self.x_max + self.x_min) / 2.

    @property
    def ycenter(self):
        """Grid center in the y-direction"""
        return (self.y_max + self.y_min) / 2.

    @property
    def dx(self):
        """Grid spacing in the x-direction"""
        return (self.x_max - self.x_min) / (self.Nx - 1)  # Denominator - 1 because endpoint=True in xs

    @property
    def dy(self):
        """Grid spacing in the y-direction"""
        return (self.y_max - self.y_min) / (self.Ny - 1)  # Denominator - 1 because endpoint=True in ys

    @property
    def dz(self):
        """Grid spacing in the z-direction"""
        return (self.z_min - self.z_max) / (self.Nz - 1)  # Denominator - 1 because endpoint=True in zs

    @property
    def xs(self):
        """Gridpoints in the x-direction"""
        xs = np.linspace(self.x_min, self.x_max, num=self.Nx, endpoint=True)
        return xs

    @property
    def ys(self):
        """Gridpoints in the y-direction"""
        ys = np.linspace(self.y_min, self.y_max, num=self.Ny, endpoint=True)
        return ys

    @property
    def xy(self):
        """Meshgrid of the horizontal gridpoints"""
        x, y = np.meshgrid(self.xs, self.ys, indexing='ij')
        return x, y

    @property
    def zs(self):
        """Gridpoints in the z-direction"""
        zs = np.linspace(self.z_min, self.z_max, num=self.Nz, endpoint=True)
        return zs

    @property
    def shape(self):
        """Grid shape"""
        return self.Nx, self.Ny, self.Nz

    @property
    def locations(self):
        """All the gridpoints of this SubGrid object, arranged as an array of 3D coordinates."""
        # SubGrid arrays
        xs = self.xs
        ys = self.ys
        zs = self.zs
        # Meshgrid
        x_m, y_m, z_m = np.meshgrid(xs, ys, zs, indexing="ij")
        # Ravel
        x_locs = np.ravel(x_m)
        y_locs = np.ravel(y_m)
        z_locs = np.ravel(z_m)
        # Combine into single array
        locations = np.stack([x_locs, y_locs, z_locs], axis=1)
        return locations


class WakeModelVelocityHandler:
    """
    Class of objects to calculate and store the wake model velocity on high-resolution grids in a small region around
    wind farms.

    Some parametrizations, such as DispersiveStresses, require evaluating the wake model velocity on a dense grid around
    the wind farm. However, this is a very expensive calculation, that should be avoided as much as possible. Therefore,
    to simplify the basic Coupling classes, the implementations related to evaluating and storing the wake model
    velocities on a dense 3D grid are put into this separate class.
    """

    def __init__(self, grid_ratio):
        """
        Initialize a WakeModelVelocityHandler object.

        Parameters
        ----------
        grid_ratio  float
            The ratio of the average turbine diameter to the spacing of the 3D grid
        """
        self._grid_ratio = grid_ratio
        # Computation results to be stored later
        self._subgrid = None
        self._u_sg = None
        self._v_sg = None

    @property
    def grid_ratio(self):
        """The ratio of the average turbine diameter to the spacing of the 3D grid"""
        return self._grid_ratio

    @property
    def subgrid(self):
        """Subgrid used by this WakeModelVelocityHandler object"""
        return self._subgrid

    @property
    def u_sg(self):
        """Wake model velocity in the x-direction"""
        return self._u_sg

    @property
    def v_sg(self):
        """Wake model velocity in the y-direction"""
        return self._v_sg

    def set_up_subgrid(self, wind_farm, abl):
        """Create a SubGrid object for the given wind farm."""
        # SubGrid spacing
        D_t = np.mean([turb.D for turb in wind_farm.turbines])
        d = D_t / self.grid_ratio
        # Set up grid dimensions
        x_min, x_max, y_min, y_max = wind_farm.coupling.region_around_farm(wind_farm)
        z_min = abl.zs[0]       # z_min = zs[0], so that integration is consistent with abl setup
        z_max = 1.6 * abl.H1    # We take z_max=1.6*H1 for now, assuming the eta perturbations stay below that value
        # Number of gridpoints
        Nx = int((x_max - x_min) / d) + 1
        Ny = int((y_max - y_min) / d) + 1
        Nz = int((z_max - z_min) / d) + 1
        # Create subgrid
        sg = SubGrid(Nx, Ny, Nz, x_min, x_max, y_min, y_max, z_min, z_max)
        # Store object
        self._subgrid = sg

    def calculate_u_wm(self, wind_farm, abl, wake_model, u_bg_evaluator, apm_evaluator):
        """Evaluate and store the wake model velocities on a dense 3D grid."""
        # Set up subgrid
        self.set_up_subgrid(wind_farm, abl)
        # Get velocity field through wake model
        u_wm, v_wm = wake_model.get_u_subgrid(wind_farm, abl, u_bg_evaluator, apm_evaluator, self.subgrid)
        # Store results
        self._u_sg = u_wm
        self._v_sg = v_wm


class SelfSimilarWMVH(WakeModelVelocityHandler):
    """
    WakeModelVelocityHandler class optimized for the self-similar wake merging method of Lanzilao and Meyers [1].

    When using the self-similar wake merging method, the wake shape functions W can be calculated separately. If the
    turbine directions and thrust coefficients do not change when the background velocity is updated, the shape
    functions don't have to be re-calculated, which significantly reduces the computational cost. This class uses that,
    speeding up the model.

    References
    ----------
    .. [1] Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the presence
    of heterogeneous background velocity fields. Wind Energy, 25(2), 237–259. https://doi.org/10.1002/we.2669
    """

    def __init__(self, grid_ratio):
        """
        Initialize a SelfSimilarWMVH object.

        Parameters
        ----------
        grid_ratio  float
            The ratio of the average turbine diameter to the spacing of the 3D grid
        """
        super().__init__(grid_ratio)
        # Computation results to be stored later
        self._w = None
        self._w_state = None

    @property
    def w(self):
        """Wake shape functions"""
        return self._w

    @property
    def w_state(self):
        """Dict of variables associated with the last calculation of w"""
        return self._w_state

    def calculate_u_wm(self, wind_farm, abl, wake_model, u_bg_evaluator, apm_evaluator):
        """Evaluate and store the wake model velocities on a dense 3D grid."""
        # Check if w has to be re-calculated, based on the new values for Ct and et
        Ct = wind_farm.coupling.Ct
        et = wind_farm.coupling.et
        new_state = {"Ct": Ct,
                     "et": et}
        necessary = self.calculation_necessary(new_state)
        # If necessary, compute w
        if necessary:
            # Update state associated with w calculation
            self._w_state = new_state
            # Set up subgrid
            self.set_up_subgrid(wind_farm, abl)
            # Calculate the wake deficit factor through wake model
            self._w = wake_model.wake_deficit(wind_farm, abl, u_bg_evaluator, apm_evaluator, self.subgrid)
        # Apply wake multiplier to background velocity
        u_wm, v_wm = wake_model.apply_wake_multiplier(wind_farm, abl, u_bg_evaluator, self.subgrid, self._w)
        # Store results
        self._u_sg = u_wm
        self._v_sg = v_wm

    def calculation_necessary(self, new_state):
        """Determine whether the stored wake shape functions have to be updated. If the turbine directions and thrust
        coefficients have not changed, the stored wake functions do not have to be re-calculated."""
        # Check if w has been calculated before
        if self.w is None:
            return True
        # Read out old state
        Ct0 = self.w_state["Ct"]
        et0 = self.w_state["et"]
        # Read out new state
        Ct = new_state["Ct"]
        et = new_state["et"]
        # Determine if w has to be re-calculated
        tolerance = 1.e-3
        necessary = (norm(Ct - Ct0) / len(Ct0) > tolerance or
                     norm(et - et0) / len(Ct0) > tolerance)
        return necessary
