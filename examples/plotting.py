"""
A file containing plotting tools commonly used in the APM.

This automates some common visualizations, so that basic scripts can be set up easily.
"""

__author__ = "Devesse Koen"
__date__ = "July 2, 2021"

import numpy as np
import matplotlib as mpl
from matplotlib.colors import CenteredNorm

# Increase font size for readability
font_size = 16
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['axes.labelsize'] = font_size
mpl.rcParams['legend.fontsize'] = font_size


def plot_field(field, grid, label, ax, f, x_lims, y_lims):
    """
    Plots a variable field of the APM.

    Only a section of the field is plotted, which results in a significant speedup.

    Mandatory parameters:
    :param field:   2D array
                        Variable that is plotted (eg. velocity perturbation, inversion displacement, ...)
    :param grid:    Grid object
                        Numerical grid of the APM
    :param label:   str
                        Label of the color bar
    :param ax:      Axes object
                        Axes object on which this function plots
    :param f:       Figure object
                        Figure object on which this function plots
    :param x_lims:  array-like
                        Plot limits for the x-axis
    :param y_lims:  array-like
                        Plot limits for the y-axis
    """
    # Select relevant part of the grid
    select_x = np.logical_and(x_lims[0] <= grid.xs/1.e3, grid.xs/1.e3 <= x_lims[1])
    select_y = np.logical_and(y_lims[0] <= grid.ys/1.e3, grid.ys/1.e3 <= y_lims[1])
    # Plot
    im = ax.pcolormesh(
        grid.xs[select_x] / 1.0e3,
        grid.ys[select_y] / 1.0e3,
        np.transpose(field[select_x][:, select_y]),
        shading='gouraud',
        cmap='RdBu_r',
        norm=CenteredNorm(),
        rasterized=True)
    # Add colorbar
    cbar = f.colorbar(im, ax=ax, shrink=1.0)
    cbar.set_label(label)
    # Axes limits and aspect ratio
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    # Axes labels
    ax.set_xlabel(r'$x\;[\mathrm{km}]$')
    ax.set_ylabel(r'$y\;[\mathrm{km}]$')
    return


def apm_plot(fields, grid, abl, etas, wave_field, z_upper, ax, x_lims,
             Nz=100, eta_0=None, forcing=None):
    """
    Plots an x-z plane through the center of the given fields, visualizing the vertical structure of the APM.

    Mandatory parameters:
    param fields:   Array-like of 2D arrays
                        Array of variables that are plotted (eg. velocity perturbation, pressure perturbation, ...)
                        The array consists of 2 2d arrays, corresponding to the variable in the two APM layers.
    param grid:     Grid object
                        Numerical grid of the APM
    param abl:      Forcing object
                        ABL object of the APM
    param etas:     Array-like of 2D arrays
                        Array of 2 2d arrays, corresponding to the thickness variations of both ABL layers.
    param ax:       Axes object
                        Axes object on which this function plots
    param x_lims:   array-like
                        Plot limits for the x-axis

    Optional parameters:
    param Nz:       int
                        Number of plotted points in the vertical direction within the ABL
    param eta_0:    np.array
                        2d array containing the topography over the domain
    param forcing:  CST or WindFarm object
                        Region of this forcing term will be indicated on the plot
    """
    # Set up x-vector
    full_xvector = grid.xs/1.e3
    if x_lims is None:
        x_lims = [full_xvector[0], full_xvector[-1]]
    select_x = np.logical_and(x_lims[0] <= full_xvector, full_xvector <= x_lims[1])
    xvector = full_xvector[select_x]
    Nx = len(xvector)
    # Set up function to get field variables along centerline
    def get_centerline(field):
        variable = field[:, int(grid.Ny/2)]
        return variable[select_x]
    # Set up function to get wave variable along centerline
    def get_centerplane(field):
        variable = field[:, int(grid.Ny/2), :]
        return variable[select_x, :]
    # Get variables along center line
    variables = np.zeros((2, Nx))
    for i in range(2):
        variables[i, :] = get_centerline(fields[i])
    wave_slice = get_centerplane(wave_field)
    # Define 2D (x-z) field
    zvector = np.concatenate([np.linspace(0., abl.H/1.e3, Nz), z_upper/1.e3])
    slice = np.zeros((Nx, len(zvector)))
    # Define layer boundaries
    if eta_0 is None:
        eta_0 = np.zeros(grid.shape)
    H0 = get_centerline(eta_0) / 1.0e3
    H1 = (abl.H1 + get_centerline(eta_0+etas[0])) / 1.0e3
    H2 = (abl.H + get_centerline(eta_0+etas[0]+etas[1])) / 1.0e3
    # Fill in 2D field
    for i in range(Nx):
        slice[i, H0[i] < zvector] = variables[0, i]
        slice[i, H1[i] < zvector] = variables[1, i]
        slice[i, max(abl.H/1.e3, H2[i]) < zvector] = wave_slice[i, max(abl.H/1.e3, H2[i]) < z_upper/1.e3]
        slice[i, np.logical_and(H2[i] < zvector, zvector <= abl.H/1.e3)] = wave_slice[i, 0]  # Default to plotting wave perturbation at abl.H for eta_t<0
    # Plot 2D field
    im = ax.pcolormesh(
        xvector,
        zvector,
        np.transpose(slice),
        shading='gouraud',
        norm=CenteredNorm(),
        cmap='RdBu_r',
        rasterized=True)
    # Plot layer boundaries
    ax.plot(xvector, np.zeros(Nx), '-k')
    ax.plot(xvector, H0, '-k')
    ax.plot(xvector, H1, '-k')
    ax.plot(xvector, H2, '-k')
    # Plot forcing terms
    if forcing is not None:
        # Define region
        x1 = np.min(forcing.vertices[:, 0])/1.e3
        x2 = np.max(forcing.vertices[:, 0])/1.e3
        # Lower layer height at region boundaries
        h1 = np.interp(x1, xvector, H1)
        h2 = np.interp(x2, xvector, H1)
        # Plot region
        ax.plot([x1, x1], [0., h1], ':k')
        ax.plot([x2, x2], [0., h2], ':k')
    # Set labels
    ax.set_xlabel(r'$x\;[\mathrm{km}]$')
    ax.set_ylabel(r'$z\;[\mathrm{km}]$')
    # Plot limits
    ax.set_xlim(x_lims)
    return im
