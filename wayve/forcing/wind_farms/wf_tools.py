#!/usr/bin/env python

'''
Some additional functions for WindFarm calculations
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__   = "June 22, 2023"


import numpy as np
import numba
from numba import njit
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull


@njit(parallel=False)
def footprint(xs, ys, dx, dy, L, xloc, yloc):
    '''
    Wind Turbine footprint after filtering in Numba syntax.
    This default method uses a Gaussian filter.

    Limit denotes the size of the domain which contains the wind Turbine footprint.
    Only the row,column and value of the footprint are stored, to limit memory usage
    and to speed up the footprint evaluation.
    Possible improvements:
        -express limit as function of the grid resolution

    Parameters
    ----------
    xs,ys: 1d numpy array
        x and y points on numerical domain
    dx,dy: 1d numpy array
        grid cell size
    xloc,yloc: scalar
        x and y location of the individual Turbine

    Returns
    -------
    row: 1d numpy array
        row index of wind Turbine footprint
    col: 1d numpy array
        column index of wind Turbine footprint
    val: 1d numpy array
        value of wind Turbine footprint
    '''
    footprint_length = L * 5  # length of square around wind Turbine stored in footprint matrix [m]
    limit_x = int(footprint_length / dx)
    limit_y = int(footprint_length / dy)
    limit = max(limit_x, limit_y)
    if dx > 400 or dy > 400:
        limit = 13  # safe value to store the whole footprint - can be reduced considerably if dx,dy large
    R = np.zeros((limit * 2, limit * 2))
    val = np.zeros(((limit * 2) ** 2))
    row = np.zeros(((limit * 2) ** 2), dtype=np.int64)
    col = np.zeros(((limit * 2) ** 2), dtype=np.int64)
    summ = 0
    count = 0
    plus_i = int((xloc - xs[0]) / dx) - limit
    plus_j = int((yloc - ys[0]) / dy) - limit
    for i in numba.prange(limit * 2):
        for j in numba.prange(limit * 2):
            R[i, j] = 1. / (np.pi * L ** 2) * np.exp(
                -((xs[plus_i + i] - xloc) ** 2 + (ys[plus_j + j] - yloc) ** 2) / L ** 2)
            summ = summ + R[i, j]
    for i in numba.prange(limit * 2):
        for j in numba.prange(limit * 2):
            val[count] = R[i, j] / (summ * dx * dy)
            if val[count] < 1.e-20:
                val[count] = 0.
                count += 1
            else:
                row[count] = plus_i + i
                col[count] = plus_j + j
                count += 1
    return row[np.nonzero(val)], col[np.nonzero(val)], val[np.nonzero(val)]


@njit(parallel=False)
def evaluate_F(F0u, F0v, Ct, rotorarea, St, e_str, row, col, value):
    '''
    Compute wind-farm force on specified grid in Numba syntax

    Parameters
    ----------
    F0u,F0v: 2d numpy array
        zero-th order wind-farm force in x and y (same shape as grid)
    Ct: Float
        Turbine thrust coefficient
    rotorarea: float
        rotor swept area
    St: float
        Turbine inflow velocity
    e_str: 1d numpy array
        wind direction at the Turbine
    row: 1d numpy array
        row index of wind Turbine footprint
    col: 1d numpy array
        column index of wind Turbine footprint
    val: 1d numpy array
        value of wind Turbine footprint

    Returns
    -------
    F0u,F0v: 2d numpy array
        Wind-farm force in x and y (same shape as grid)
    '''
    for i in numba.prange(len(value)):
        F0u[row[i], col[i]] = F0u[row[i], col[i]] + (0.5 * Ct * rotorarea *
                                                     St ** 2 * e_str[0] * value[i])
        F0v[row[i], col[i]] = F0v[row[i], col[i]] + (0.5 * Ct * rotorarea *
                                                     St ** 2 * e_str[1] * value[i])
    return F0u, F0v


def wind_farm_shape(wind_farm):
    """
    Finds the smallest convex polygon that encloses the given wind farm.

    Parameters
    ----------
    wind_farm: WindFarm
        Wind farm object

    Returns
    -------
    vertices: np array
        Closed loop of the polygon vertices in counter-clockwise order, shape (npoints,2)
    area: float
        Surface area of the polygon
    """
    # Set up array of turbine coordinates
    turb_coordinates = np.stack([wind_farm.xs, wind_farm.ys], axis=-1)
    if turb_coordinates.shape[0] == 1:
        return turb_coordinates, 0.
    # Initialize ConvexHull object
    hull = ConvexHull(turb_coordinates)
    # Get edge points (in counter-clockwise order)
    vertices_indices = hull.vertices
    vertices = np.stack([wind_farm.xs[vertices_indices], wind_farm.ys[vertices_indices]], axis=-1)
    # Close loop of edge points
    vertices = np.vstack([vertices, vertices[0, :]])
    # Get surface area
    area = hull.volume
    return vertices, area


@njit(parallel=False)
def height_average(field_3d, h, z):
    """
    Return the height-average of a 3D field up to the varying altitude h.

    z should be ordered, and it's highest value should always be higher than max(h).

    Parameters
    ----------
    field_3d: 3d numpy array
        field to be height-averaged
    h: 2d numpy array
        height up to which averaging is done
    z: 1d numpy array
        vertical grid of the given 3d field

    Returns
    -------
    field_ha: 2d numpy array
        height-averaged field (same shape as h)
    """
    # Grid size
    Nx = field_3d.shape[0]
    Ny = field_3d.shape[1]
    # Set up output array
    field_ha = np.zeros((Nx, Ny))
    for i in numba.prange(Nx):
        for j in numba.prange(Ny):
            # Select array parts that are within the layer
            selection = z <= h[i, j]
            field_sel = field_3d[i, j, :]
            field_sel = field_sel[selection]
            z_sel = z[selection]
            # Interpolate to layer boundary
            final_z = h[i, j]
            final_f = np.interp(h[i, j], z, field_3d[i, j, :])  # Standard linear interpolation
            # Extend array to layer boundary
            z_sel = np.append(z_sel, final_z)
            field_sel = np.append(field_sel, final_f)
            # Trapezoidal integration (no edge correction)
            field_ha[i, j] = np.trapz(field_sel, x=z_sel) / (z_sel[-1] - z_sel[0])
    return field_ha


def filter_3d(field_3d, subgrid, Lf, x_c, y_c, zero_edge=True, use_scipy=True):
    """
    Filter the given 3d field onto the coarser horizontal grid resolution.

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
    zero_edge: boolean (optional)
        Whether the field is taken to be zero outside the domain bounds, or extrapolated (default: True)
    use_scipy: boolean (optional)
        Whether a scipy or numba implementation is used. The scipy implementation is faster, but does not allow for
        non-uniform grids (default: True)
    """
    # Output shape
    Nx = len(x_c)
    Ny = len(y_c)
    Nz = subgrid.Nz
    # Output array
    output = np.zeros((Nx, Ny, Nz))
    # Loop over heights
    for k in range(Nz):
        output[:, :, k] = filter_2d(field_3d[:, :, k], subgrid, Lf, x_c, y_c, zero_edge, use_scipy)
    return output


def filter_2d(field_2d, subgrid, Lf, x_c, y_c, zero_edge=True, use_scipy=True):
    """
    Filter the given 2d field onto the coarser grid resolution.

    Parameters
    ----------
    field_2d: array-like
        2D field to be filtered
    subgrid: SubGrid object
        Fine 3D grid
    Lf: float
        Filter length
    x_c: array-like
        Coarse resolution x-coordinates
    y_c: array-like
        Coarse resolution y-coordinates
    zero_edge: boolean (optional)
        Whether the field is taken to be zero outside the domain bounds, or extrapolated (default: True)
    use_scipy: boolean (optional)
        Whether a scipy or numba implementation is used. The scipy implementation is faster, but does not allow for
        non-uniform grids (default: True)
    """
    # Set up subgrid arrays
    xs = subgrid.xs
    ys = subgrid.ys
    if zero_edge and not use_scipy:
        # Set up meshgrid
        x_m, y_m = subgrid.xy
        # Gaussian filter
        return filter_2d_numba(field_2d, x_c, y_c, xs, ys, x_m, y_m, Lf)
    # Use scipy implementation
    return filter_2d_scipy(field_2d, x_c, y_c, xs, ys, Lf, zero_edge=zero_edge)


def filter_2d_scipy(field_2d, x_c, y_c, xs, ys, L_f, zero_edge=True):
    # Grid spacing
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    # Filter settings
    sigma_x = L_f / (np.sqrt(2.) * dx)
    sigma_y = L_f / (np.sqrt(2.) * dy)
    sigmas = [sigma_x, sigma_y]
    mode = 'nearest'
    if zero_edge:
        mode = 'constant'
    # Filtered field
    filt = gaussian_filter(field_2d, sigmas, mode=mode)
    # Interpolation function
    fill_value = None
    if zero_edge:
        fill_value = 0.
    f_int = RegularGridInterpolator((xs, ys), filt,
                                    method="linear", bounds_error=False, fill_value=fill_value)
    # Evaluate on coarse grid
    x_m, y_m = np.meshgrid(x_c, y_c, indexing="ij")
    avg_space = f_int((x_m, y_m))
    return avg_space


@njit(parallel=True)
def filter_2d_numba(field_2d, x_c, y_c, xs, ys, xm, ym, L_f):
    """
    Filter the given 2d field onto the coarser grid resolution.

    This method allows for non-uniform grids.
    This method assumes the given field is zero beyond the grid boundaries.
    """
    # Number of c points
    Nx_c = len(x_c)
    Ny_c = len(y_c)
    # Gaussian filter
    avg_space = np.zeros((Nx_c, Ny_c))
    for i in numba.prange(Nx_c):
        for j in numba.prange(Ny_c):
            # Select region of grid around c point
            min_i = np.argmin(np.abs(xs-(x_c[i]-5*L_f)))
            min_j = np.argmin(np.abs(ys-(y_c[j]-5*L_f)))
            max_i = np.argmin(np.abs(xs-(x_c[i]+5*L_f)))
            max_j = np.argmin(np.abs(ys-(y_c[j]+5*L_f)))
            xs_c = xs[min_i:max_i]
            ys_c = ys[min_j:max_j]
            xm_c = xm[min_i:max_i, min_j:max_j]
            ym_c = ym[min_i:max_i, min_j:max_j]
            fm_c = field_2d[min_i:max_i, min_j:max_j]
            # Multiply with Gaussian kernel
            gfilter = filter_function_mesh(xm_c, ym_c, x_c[i], y_c[j], L_f)
            multiplied = gfilter * fm_c
            # Integration
            over_y = np.zeros(len(xs_c))
            for x_ind in numba.prange(len(xs_c)):
                over_y[x_ind] = np.trapz(multiplied[x_ind, :], x=ys_c)
            avg_space[i, j] = np.trapz(over_y, x=xs_c)
    return avg_space


@njit(parallel=False)
def filter_function_mesh(x, y, grid_xloc, grid_yloc, L_f):
    '''
    Spatial filter kernel
    xm, ym form a mesh
    '''
    dist = (x - grid_xloc) ** 2 + (y - grid_yloc) ** 2
    R = 1. / (np.pi * L_f ** 2) * np.exp(-dist / L_f ** 2)
    return R
