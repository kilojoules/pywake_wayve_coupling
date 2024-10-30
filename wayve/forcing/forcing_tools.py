'''
Low-level helper functions for calculations related to the APM forcing parametrizations
'''

__author__ = "Dries Allaerts, Koen Devesse"
__date__ = "April 9, 2024"


import numpy as np
from numba import njit


@njit
def e_streamwise(Uinf, Vinf):
    '''Unit vector along the wind direction'''
    Sinf = np.sqrt(Uinf ** 2 + Vinf ** 2)
    return np.array([Uinf, Vinf]) / Sinf


@njit
def e_spanwise(Uinf, Vinf):
    '''Unit vector in cross wind direction'''
    Sinf = np.sqrt(Uinf ** 2 + Vinf ** 2)
    return np.array([-Vinf, Uinf]) / Sinf


def fill_shape(vertices, x, y):
    """
    Set the region of the given grid (x, y) within the given polygon to 1, and the remainder to 0.

    This implementation only works for convex shapes. The points describing the vertices of the polygon should be in
    counter-clockwise order, and should form a closed loop, so that the first and last points are the same.

    Parameters
    ----------
    vertices: np array
        Vertices of the polygon as a closed loop, ordered counter-clockwise, shape (npoints,2)
    x: np array
        x coordinates (mesh)
    y: np array
        y coordinates (mesh)
    """
    # Number of vertices
    grid_shape = x.shape
    n_v = vertices.shape[0]
    # No polygon
    if n_v <= 1:
        return np.zeros(grid_shape)
    # Set up output array
    out = np.ones(grid_shape)
    # Loop over the shape edges
    for i in range(n_v-1):
        # Get current vertex
        vert_x = vertices[i, 0]
        vert_y = vertices[i, 1]
        # Get edge
        edge_dx = vertices[i+1, 0] - vertices[i, 0]     # x-component of edge vector
        edge_dy = vertices[i+1, 1] - vertices[i, 1]     # y-component of edge vector
        # Get inward-pointing vector (vertices ordered counter-clockwise)
        vec_in = np.array([-edge_dy, edge_dx])
        # Compute distance from edge (inwards being positive, outwards negative)
        dist = vec_in[0] * (x-vert_x) + vec_in[1] * (y-vert_y)
        # Remove region to the outside of this edge
        out = np.multiply(out, np.heaviside(dist, 0.5))
    return out
