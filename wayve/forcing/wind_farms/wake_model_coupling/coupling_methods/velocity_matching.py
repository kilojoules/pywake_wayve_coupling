import numba
import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import VaryingBackground,\
    filtered_in_contribution, height_average_shape_function


class VelocityMatching(VaryingBackground):
    """
    A Coupling class that implements the Velocity Matching method from Devesse et al. [1].

    This formulation is built around using the self-similar uni-directional wake merging method from Lanzilao and Meyers
    [2].

    References
    ----------
    .. [1]  Devesse, K., Lanzilao, L., & Meyers, J. (2023). A meso-micro atmospheric perturbation model for wind farm
    blockage. Preprint. http://arxiv.org/abs/2310.18748
    .. [2]  Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the presence
    of heterogeneous background velocity fields. Wind Energy, 25(2), 237–259. https://doi.org/10.1002/we.2669
    """

    def __init__(self, wake_model, wm_velocity_evaluator, shape_frac=1.):
        """
        Initialize a VelocityMatching object

        Parameters
        ----------
        wake_model  UniDirectionalSelfSimilar object
                        Wake model that this Coupling interfaces to.
                        This implementation requires the wake model to be uni-directional, and based on the wake merging
                        method from Lanzilao and Meyers [1].
        wm_velocity_handler     SelfSimilarWMVH object
                        Wake model handler used by this coupling
        shape_frac  float
                        Ratio of the filter length to the shape function spacing in each direction

        References
        ----------
        .. [1]  Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the
        presence of heterogeneous background velocity fields. Wind Energy, 25(2), 237–259.
        https://doi.org/10.1002/we.2669
        """
        super().__init__(wake_model, wm_velocity_evaluator)
        self.__shape_frac = shape_frac

    @property
    def shape_frac(self):
        """Ratio of the filter length to the shape function spacing in each direction"""
        return self.__shape_frac

    @shape_frac.setter
    def shape_frac(self, value):
        self.__shape_frac = value

    def update_ub_vb(self, model, wind_farm, result):
        """
        Update the evaluator functions for u_b and v_b, based on the given APM state.

        As this Coupling explicitly uses the uni-directional merging method from Lanzilao and Meyers [1], the velocity
        matching equation only has to be solved for the streamwise direction. Therefore, the two-directional velocity
        field is split up into a main streamwise direction, for which the background velocity is calculated, and a
        spanwise direction, where the background velocity is not relevant.

        References
        ----------
        .. [1]  Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the
        presence of heterogeneous background velocity fields. Wind Energy, 25(2), 237–259.
        https://doi.org/10.1002/we.2669
        """

        # APM components
        abl = model.abl
        grid = model.grid

        # Get subgrid
        subgrid = self.wm_velocity_handler.subgrid

        # Get altitude array
        z = subgrid.zs

        # # Split up into streamwise and spanwise components # #

        # Get Turbine direction (streamwise)
        e_str, e_span = self.wake_model.background_flow_direction(wind_farm, abl)
        theta_str_inf = np.arctan2(e_str[1], e_str[0])  # Angle in the wind direction

        # Background velocity #

        # Read out x-, y-components from ABL
        u_abl = abl.u(z)
        v_abl = abl.v(z)
        # Streamwise component
        str_abl = u_abl * e_str[0] + v_abl * e_str[1]

        # APM velocity #

        # Read out u, v from APM state
        u_apm = abl.U1 + result["u1r"]
        v_apm = abl.V1 + result["v1r"]
        # Streamwise component
        str_apm = u_apm * e_str[0] + v_apm * e_str[1]

        # # Get wake function multiplier # #
        w = self.wm_velocity_handler.w

        # # Solve for background velocities # #

        # Solve streamwise component
        str_b = self.solve_streamwise(wind_farm, abl, grid, str_abl, str_apm, w)

        # Solve spanwise component
        span_b = self.solve_spanwise(wind_farm, abl, subgrid, e_span)

        # # Convert back to u and v components # #
        u_b = str_b * e_str[0] + span_b * e_span[0]  # Resulting x-component of the velocity
        v_b = str_b * e_str[1] + span_b * e_span[1]  # Resulting y-component of the velocity

        # Get shape functions
        x_shape, y_shape, dx, dy = self.shape_function_setup(subgrid, wind_farm.Lfilter)

        # Interpolation functions
        f_ub = RegularGridInterpolator((x_shape, y_shape), u_b, method="linear", bounds_error=False, fill_value=None)
        f_vb = RegularGridInterpolator((x_shape, y_shape), v_b, method="linear", bounds_error=False, fill_value=None)

        # Set up evaluator functions
        self.ub_evaluator = lambda x, y: f_ub((x, y))
        self.vb_evaluator = lambda x, y: f_vb((x, y))

    def solve_spanwise(self, wind_farm, abl, subgrid, e_span):
        """
        Solves the velocity matching equation for the spanwise direction.

        The code assumes there are no wakes in the spanwise direction, simplifying the matching equation to scaling the
        APM velocity with the background profile shape function.
        """
        # Get shape functions
        x_shape, y_shape, dx, dy = self.shape_function_setup(subgrid, wind_farm.Lfilter)
        x_m, y_m = np.meshgrid(x_shape, y_shape, indexing='ij')
        shape_s = (len(x_shape), len(y_shape))
        x_locs = np.ravel(x_m)
        y_locs = np.ravel(y_m)
        # Evaluate apm state at shape function centers
        u_apm_r, v_apm_r, eta_r = self.apm_evaluator(x_locs, y_locs)
        u_apm = np.reshape(u_apm_r, shape_s)
        v_apm = np.reshape(v_apm_r, shape_s)
        eta = np.reshape(eta_r, shape_s)
        h1 = abl.H1 + eta
        # Get spanwise perturbations
        span_0 = abl.U1 * e_span[0] + abl.V1 * e_span[1]
        span_t = u_apm * e_span[0] + v_apm * e_span[1]
        span_p = span_t - span_0
        # Vertical grid and shape function
        z = subgrid.zs
        f = self.vertical_profile(abl, z)
        # Height-averaged shape function
        f_ha = height_average_shape_function(f, z, h1)
        # Get background velocity scales u_b and v_b
        span_b = np.divide(span_p, f_ha)
        return span_b

    def solve_streamwise(self, wind_farm, abl, grid, str_abl, str_apm, w):
        """
        Solves the velocity matching equation for the streamwise direction used by the wake model.
        """

        # Filter length
        L_f = wind_farm.Lfilter

        # Get subgrid
        subgrid = self.wm_velocity_handler.subgrid
        xs = subgrid.xs
        ys = subgrid.ys
        xm, ym = subgrid.xy
        z = subgrid.zs

        # Set up shape function and collocation points
        x_col, y_col = self.collocation_point_setup(grid, subgrid)
        Nx_col = len(x_col)
        Ny_col = len(y_col)
        x_shape, y_shape, dx, dy = self.shape_function_setup(subgrid, wind_farm.Lfilter)
        Nx_shape = len(x_shape)
        Ny_shape = len(y_shape)

        # Add background shape
        shape = self.vertical_profile(abl, z)
        w_prod = w * shape

        # Height-average
        w_prod = self.exact_height_average(w_prod, subgrid)

        # # # Set up linear system # # #

        # Set up matrix
        A = VelocityMatching.set_up_matrix(Nx_col, Ny_col, Nx_shape, Ny_shape,
                                           x_col, y_col, x_shape, y_shape,
                                           dx, dy,
                                           xs, ys, xm, ym,
                                           L_f, w_prod)

        # Set up RHS
        b = self.set_up_rhs(wind_farm, grid, subgrid, x_col, y_col, str_abl, str_apm, w)

        # Solve system
        A_pinv = np.linalg.pinv(A)
        str_b = np.matmul(A_pinv, b)

        # Reshape output to expected shape
        str_b = np.reshape(str_b, (Nx_shape, Ny_shape))

        return str_b

    def collocation_point_setup(self, grid, subgrid):
        """Set up the collocation points"""
        x_col, y_col = subgrid.in_common(grid)
        return x_col, y_col

    def eval_at_collocation_points(self, grid, subgrid, var):
        """Evaluate a given variable on the collocation points"""
        var_col = subgrid.in_domain(var, grid)
        return var_col

    def shape_function_setup(self, subgrid, L_f):
        """Set up the shape functions. These are defined by their centers and their spacing."""
        # Region bounds
        x_min = subgrid.x_min
        x_max = subgrid.x_max
        y_min = subgrid.y_min
        y_max = subgrid.y_max
        # Number of shape functions
        Nx_shape = max(2, int(self.shape_frac * (x_max-x_min) / L_f))
        Ny_shape = max(2, int(self.shape_frac * (y_max-y_min) / L_f))
        # Shape function setup
        x_shape = np.linspace(x_min, x_max, Nx_shape, endpoint=True)
        y_shape = np.linspace(y_min, y_max, Ny_shape, endpoint=True)
        # Spacing
        dx = x_shape[1]-x_shape[0]
        dy = y_shape[1]-y_shape[0]
        return x_shape, y_shape, dx, dy

    def set_up_rhs(self, wind_farm, grid, subgrid, x_col, y_col, str_abl, str_apm, w):
        """Set up the right-hand side of the velocity matching equation"""
        # Unperturbed ABL term #
        # Background velocity
        s_abl_sg = np.empty(w.shape)
        s_abl_sg[:, :] = str_abl
        # Add wakes
        s_abl_sg = np.multiply(s_abl_sg, w)
        # Height-average and filter
        abl_term = self.ha_and_filter(s_abl_sg, subgrid, wind_farm.Lfilter, x_col, y_col,
                                      zero_edge=True, use_scipy=False)
        # WindFarm APM term #
        # Only use apm field within subgrid domain
        apm_term = self.eval_at_collocation_points(grid, subgrid, str_apm)
        # APM filtered-in term #
        # Get filtered in contribution
        filt_in_term = filtered_in_contribution(grid, subgrid, str_apm, wind_farm.Lfilter, x_col, y_col)
        # Combine #
        str_rhs = (apm_term - abl_term - filt_in_term).flatten()
        return str_rhs

    @staticmethod
    def set_up_matrix(Nx_col, Ny_col, Nx_shape, Ny_shape, x_col, y_col, x_shape, y_shape, dx, dy, xs, ys, xm, ym,
                      L_f, w_prod):
        """Set up the matrix. Can set up both square and rectangular matrices."""
        # Set up matrix
        A = np.zeros((Nx_col * Ny_col, Nx_shape * Ny_shape))
        # Determine domain bounds
        x_bound = np.array([x_shape[0], x_shape[-1]])
        y_bound = np.array([y_shape[0], y_shape[-1]])
        # Gaussian filter kernel settings
        sigma_x = L_f / (np.sqrt(2.) * (xs[1]-xs[0]))
        sigma_y = L_f / (np.sqrt(2.) * (ys[1]-ys[0]))
        sigmas = [sigma_x, sigma_y]
        # Loop over shape functions
        for i in numba.prange(Nx_shape):
            # Shape function center x-coordinate
            x_i = x_shape[i]
            for j in numba.prange(Ny_shape):
                # Shape function center y-coordinate
                y_j = y_shape[j]
                # Determine matrix column
                col = i * Ny_shape + j
                # Get shape function
                phi = shape_function(xs, ys, x_i, y_j, dx, dy, x_bound, y_bound)
                # Multiply with wake modifier
                multiplied = np.multiply(phi, w_prod)
                # Filter
                filtered = gaussian_filter(multiplied, sigmas, mode='constant')
                # Set up interpolation function
                f_filt = RegularGridInterpolator((xs, ys), filtered)
                # Loop over collocation points
                for k in numba.prange(Nx_col):
                    # Test function x-coordinate
                    x_k = x_col[k]
                    for l in numba.prange(Ny_col):
                        # Test function y-coordinate
                        y_l = y_col[l]
                        # Determine matrix row
                        row = k * Ny_col + l
                        # Evaluate filtered contribution at test point
                        coeff = f_filt([x_k, y_l])
                        # Add to matrix
                        A[row, col] = coeff
        return A


@njit(parallel=False)
def shape_function(xs, ys, x_loc, y_loc, dx, dy, x_bound, y_bound):
    '''
    First-order shape function around x_loc and y_loc (peak at (x_loc,y_loc), zero at adjacent gridpoints)
    '''
    # Get grid size
    Nx = len(xs)
    Ny = len(ys)
    # Set up shape function
    phi = np.zeros((Nx, Ny))
    # Iterate over grid
    for i in numba.prange(Nx):
        x = xs[i]
        if x_bound[0] <= x <= x_bound[1]:
            for j in numba.prange(Ny):
                y = ys[j]
                if y_bound[0] <= y <= y_bound[1]:
                    # Compact support
                    if np.abs(x-x_loc) < dx and np.abs(y-y_loc) < dy:
                        phi[i, j] = (1 - np.abs(x-x_loc) / dx) * (1 - np.abs(y-y_loc) / dy)
    return phi
