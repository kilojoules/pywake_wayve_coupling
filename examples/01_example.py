#!/usr/bin/env python

"""
A basic example script that runs the APM with basic settings, based on the cases analyzed by Allaerts and Meyers (2019).

References
----------
..  Allaerts, D., & Meyers, J. (2019). Sensitivity and feedback of wind-farm-induced gravity waves. Journal of Fluid
 Mechanics, 862, 990–1028. https://doi.org/10.1017/jfm.2018.969
..  Nieuwstadt, F. T. M. (1983). On the solution of the stationary, baroclinic Ekman-layer equations with a finite
 boundary-layer height. Boundary-Layer Meteorology, 26(4), 377–390. https://doi.org/10.1007/BF00119534
"""

__author__ = "Koen Devesse, Lanzilao Luca, Dries Allaerts"
__date__ = "January 17, 2020"

import numpy as np
import matplotlib.pyplot as plt

from wayve.apm import APM
from wayve.abl.abl_setup import AM2019
from wayve.grid.grid import Stat2Dgrid
from wayve.forcing.apm_forcing import CST
from wayve.momentum_flux_parametrizations import FrictionCoefficients
from wayve.pressure.gravity_waves.gravity_waves import Uniform
from wayve.solvers import LGMRES
from wayve.pressure.gravity_waves.wave_patterns import get_du_field
from examples.plotting import apm_plot, plot_field


# ---------------------------------------------------- #
# ------------- Step 1: define abl ------------------- #
# ---------------------------------------------------- #

# The ABL object defines the background atmospheric state, and stores various variables such as the height-averaged
# velocities and the potential temperature profile.
#
# Here, we use the atmospheric conditions from the example cases used by Allaerts and Meyers (2019). Alternatively, the
# method wayve.abl.abl_setup.mesoscale_based can be used to set up an ABL object based on data typically outputted by
# mesoscale models such as COSMO-CCLM, or re-analysis data such as ERA5.

# Sub- or supercritical case
sub = True      # This determines the capping inversion strength (see Allaerts and Meyers, 2019)

# The method "AM2019" sets up the flow profiles found by Nieuwstadt (1983) with default parameters, and aligns the flow
# in the lower layer with the x-direction.
abl = AM2019(sub)


# ---------------------------------------------------- #
# -------- Step 2: define numerical grid ------------- #
# ---------------------------------------------------- #

# The Stat2Dgrid object describes the discretization of the APM, and provides methods to perform FFTs.
#
# Based on Allaerts and Meyers (2019), a grid spacing of 500m is recommended. Because of the periodic boundary
# conditions inherent to the Fourier-Galerkin spectral method, the grid should be larger than the domain of interest, so
# that the perturbations die out instead of being recycled.

# Numerical parameters
Nx = 2000           # grid points in x-direction
Lx = 1.e6           # grid size in x-direction [m]
Ny = 800            # grid points in y-direction
Ly = 0.4e6          # grid size in y-direction [m]

# The grid can be smaller in the y-direction, as in this example case the flow is aligned with the x-direction. However,
# this is not always true.

# Generate 2D grid object
grid = Stat2Dgrid(Lx, Nx, Ly, Ny)


# ----------------------------------------------------- #
# ---------- Step 3: define forcing object ------------ #
# ----------------------------------------------------- #

# The ForcingTerm object represents the perturbing element in the ABL. WAYVE provides implementations for various types
# of forcings:
#   - CST: forcing with a constant drag coefficient within a defined region
#   - WindFarm: wind farms (see 02_wind_farm.py)
#   - Topography: hills triggering mountain waves (see 03_topography.py)
#   - ForcingComposite: combines multiple forcing terms
#
# For simplicity, we use a CST object here, and loosely base the parameters on the example cases of Allaerts and Meyers
# (2019).

# CST parameters
Ct = 0.006      # Drag coefficient
length = 20.e3  # Size of the forcing region in the x-direction [m]
width = 30.e3   # Size of the forcing region in the y-direction [m]

# The CST is implemented as a convex polygon. To set up a CST object, the vertices of this polygon have to be provided
# in counter-clockwise order.
# We center the vertices around x=y=0, which corresponds to the center of the APM grid.
vertices = np.array([[-length/2, -width/2],
                     [length/2, -width/2],
                     [length/2, width/2],
                     [-length/2, width/2],
                     [-length/2, -width/2]])

# Generate forcing object
forcing = CST(Ct, vertices)


# ---------------------------------------------------- #
# ----- Step 4: create APM model from components ----- #
# ---------------------------------------------------- #

# The terms in the APM equations related to the momentum flux between the ABL layers are incorporated using an MFP
# object, with MFP standing for Momentum-Flux Parametrization.
#
# WAYVE provides three basic options:
#   - FrictionCoefficients (recommended): the parametrization used by Allaerts and Meyers (2019).
#   - PotentialFlow: parametrization representing potential flow, i.e. no momentum flux between the layers.
#   - EddyViscosity: parametrization where the momentum flux is assumed to scale directly with the velocity difference.

# In this example, we use a FrictionCoefficients object.
mfp = FrictionCoefficients()

# The pressure closure equation is incorporated using a PressureParametrization object.
# There are several options for the pressure closure equation:
#   - NoPressureFeedback: no pressure related to capping inversion displacements
#   - RigidLid: places a rigid lid at the top of the ABL
#   - GravityWaves: uses linear gravity wave theory to link the pressure feedback to the capping inversion displacement.

# In this example, we use a Uniform object (a type of GravityWaves object), which implements linear gravity wave theory
# assuming a uniform free atmosphere. We initialize the free atmosphere as hydrodynamic, but non-rotating.
pressure = Uniform(dynamic=True, rotating=False)

# Finally, we can simply initialize the APM object using all the components set up above.
model = APM(grid, forcing, abl, mfp, pressure)


# ----------------------------------------------------- #
# --------------- Step 5: solve the APM --------------- #
# ----------------------------------------------------- #

# Solver objects can use an APM object to apply linear operators (My, Px, Ax) and compute the right-hand side of the APM
# equations (Bvector).
#
# WAYVE provides three basic options:
#   - NoFeedback: direct solve, where feedback effects are not included
#   - KrylovMethod: solvers using Krylov subspace methods (only for linear forcing terms)
#   - FixedPointIteration: basic fixed point iteration solver, which includes non-linear effects

# Since the CST forcing can be linearized, we can use an LGMRES solver (a type of KrylovMethod). However, any of the
# methods mentioned above can be used.
tol = 1.e-3     # Relative tolerance threshold
max_iter = 10   # Maximum number of iterations
solver = LGMRES(tol, max_iter)

# When calling solve, the APM object performs the necessary pre-processing, and runs the solver routine. The results are
# post-processed, and stored in a dict.
verbose = True  # Flag for printing and plotting solver results
result = model.solve(solver, verbose=verbose)


# ---------------------------------------------------- #
# ----------- Some example post processing ----------- #
# ---------------------------------------------------- #

# Evaluate power extracted from the ABL. This power calculation is based on the APM momentum equations.
x = result["x"]     # APM state array
P = forcing.power(model, x)/1.0e9
print(f"Power production = {P:.2f} GW")

# Difference in velocity magnitude in wind-farm layer
M1 = -(abl.S1-np.sqrt((abl.U1+result['u1r'])**2 +
                      (abl.V1+result['v1r'])**2)) / abl.S1*100
# Difference in velocity magnitude in upper layer
M2 = -(abl.S2-np.sqrt((abl.U2+result['u2r'])**2 +
                      (abl.V2+result['v2r'])**2)) / abl.S2*100

# Field plots #
# Set up plot
f, axarr = plt.subplots(1, 3, figsize=(15., 5.))  # Make sure there is enough room for the colorbars
f.subplots_adjust(wspace=0.75)
# Plot boundaries (in km)
x_lims = [-100., 100.]
y_lims = [-100., 100.]
# Plot variables
plot_field(result['etar'] / abl.H * 100, grid, r'$\eta_t/H\;[\%]$', axarr[0], f, x_lims, y_lims)
plot_field(result['pr'] / abl.S1 ** 2 * 100, grid, r'$p^\prime/\rho_0U_1^2\;[\%]$', axarr[1], f, x_lims, y_lims)
plot_field(M1, grid, r'$u^\prime_1/U_1\;[\%]$', axarr[2], f, x_lims, y_lims)
# Plot outline of forcing term
for ax in axarr:
    ax.plot(vertices[:, 0] / 1.e3, vertices[:, 1] / 1.e3, '-k')
# Show plot
plt.show()

# APM plot #
# When using a pressure closure equation based on gravity wave theory, we can visualize the wave patterns in the free
# atmosphere. The code below produces such a plot.

# Calculate gravity wave field in the free atmosphere
Nz_upper = 100
z_upper = np.linspace(abl.H, 5.*abl.H, Nz_upper+1, endpoint=True)
print(f"Start calculations for gravity wave visualisation")
u_wave = get_du_field(result['etac'], z_upper, grid, abl, pressure) / abl.S3 * 100.
print(f"Gravity wave calculations done")
# Combine variables into input lists
etas = [result['eta1r'], result['eta2r']]
u_fields = [M1*abl.S1/abl.S3, M2*abl.S2/abl.S3]
# Set up plot
f, ax = plt.subplots()
# Perform APM plot
im = apm_plot(u_fields, grid, abl, etas, u_wave, z_upper, ax, x_lims, forcing=forcing)
# Add colorbar
cbar = f.colorbar(im, ax=ax, shrink=1.0, label=r"$\Delta u/U_g$ [%]")
# Show plot
plt.show()
