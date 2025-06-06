#!/usr/bin/env python

"""
An example script that runs the APM with a wind farm representation and wake model coupling.

The wind farm is based on the one used by Lanzilao and Meyers (2024).

References
----------
..  Lanzilao, L., & Meyers, J. (2024). A parametric large-eddy simulation study of wind-farm blockage and gravity waves
 in conventionally neutral boundary layers. Journal of Fluid Mechanics, 979, A54. https://doi.org/10.1017/jfm.2023.1088
..  Allaerts, D., & Meyers, J. (2019). Sensitivity and feedback of wind-farm-induced gravity waves. Journal of Fluid
 Mechanics, 862, 990–1028. https://doi.org/10.1017/jfm.2018.969
..  Devesse, K., Lanzilao, L., & Meyers, J. (2024). A meso-micro atmospheric perturbation model for wind farm blockage.
 Journal of Fluid Mechanics, 998, A63. https://doi.org/10.1017/jfm.2024.868
..  Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the presence of
 heterogeneous background velocity fields. Wind Energy, 25(2), 237–259. https://doi.org/10.1002/we.2669
..  Stipa, S., Ajay, A., Allaerts, D., & Brinkerhoff, J. (2024). The Multi-Scale Coupled Model: a New Framework
 Capturing Wind Farm-Atmosphere Interaction and Global Blockage Effects. Wind Energy Science, 9, 1123–1152.
 https://doi.org/10.5194/wes-9-1123-2024
..  Devesse, K., Stipa, S., Brinkerhoff, J. , Allaerts, D., & Meyers, J. (2024). Comparing methods for coupling wake
 models to an atmospheric perturbation model in WAYVE. Journal of Physics: Conference Series, 2767, 092079.
 https://doi.org/10.1088/1742-6596/2767/9/092079
"""

__author__ = "Koen Devesse"
__date__ = "October 11, 2023"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from wayve.apm import APM
from wayve.abl.abl_setup import AM2019
from wayve.grid.grid import Stat2Dgrid
from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.velocity_matching import VelocityMatching
from wayve.forcing.wind_farms.wake_model_coupling.wake_models.lanzilao_merging import Lanzilao
from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import SelfSimilarWMVH
from wayve.forcing.wind_farms.wind_farm import WindFarm, Turbine
from wayve.forcing.wind_farms.dispersive_stresses import DispersiveStresses
from wayve.forcing.wind_farms.entrainment import ConstantFlux
from wayve.forcing.apm_forcing import ForcingComposite
from wayve.momentum_flux_parametrizations import FrictionCoefficients
from wayve.pressure.gravity_waves.gravity_waves import Uniform
from wayve.solvers import FixedPointIteration

# ----------------------------------------------------- #
# ------------- Step 1: set up turbines --------------- #
# ----------------------------------------------------- #

# We use the same turbines and farm layout as Lanzilao and Meyers (2024).
D = 198.            # Turbine diameter [m]
zh = 119.           # Turbine hub height [m]
Ntx = 16            # Number of turbine rows
Nty = 10            # Number of turbine columns
Nt = Ntx * Nty      # Number of turbines
Ct = 0.88           # Turbine thrust coefficient
sx = 5 * D          # Turbine spacing in the x-direction [m]
sy = 5 * D          # Turbine spacing in the y-direction [m]
Lwfx = Ntx * sx     # Wind-farm length
Lwfy = Nty * sy     # Wind-farm width

# Turbine coordinates
xs = np.linspace(0, Lwfx, Ntx, endpoint=False)
ys, dy = np.linspace(-Lwfy/2, Lwfy/2, Nty, endpoint=False, retstep=True)
X_grid, Y_grid = np.meshgrid(xs, ys)
X = np.ravel(X_grid)
Y = np.ravel(Y_grid)
# Staggered layout
Y[1::2] += dy/2.

# The wind farm is initialized using a list of Turbine objects, which we set up using the above parameters.
turbines = []
for t in range(Nt):
    # Each turbine object is initialized with its location (X[t], Y[t]), its diameter, its hub height, and its thrust
    # coefficient. This thrust coefficient can be a constant number, or a callable that takes wind speed as an input.
    turbine = Turbine(X[t], Y[t], D, zh, Ct)
    turbines.append(turbine)


# ---------------------------------------------------- #
# ------- Step 2: set up wake model coupling --------- #
# ---------------------------------------------------- #

# The APM can not represent the individual turbine wakes. Therefore, it has to be coupled to a wake model when
# simulating wind farms. The basic Coupling expected by the APM is defined in
# wayve.forcing.wind_farms.wake_model_coupling.coupling. At its core, it simply has to provide the APM with the inflow
# velocities, directions, and thrust coefficients of all the turbines. When implementing a new coupling, it suffices to
# implement the interface defined there.
#
# WAYVE provides three coupling methods, which all implement the basic VaryingBackground class. This is a class of
# Coupling objects that calculate a background velocity based on the APM state.
# The coupling methods currently implemented in WAYVE are:
#   - Upstream: takes the velocity upstream of the farm as the background velocity
#               see Allaerts and Meyers (2019)
#   - VelocityMatching: calculates the background velocity by matching the velocity fields of the APM and the wake model
#               see Devesse et al. (2024)
#   - PressureBased: uses the pressure component of the APM velocity perturbation as the background velocity
#               see Stipa et al. (2024)
#
# For a comparison of these coupling methods, see Devesse et al. (2024b).
#
# The wake model interface expected by these coupling methods is defined in
# wayve.forcing.wind_farms.wake_model_coupling.wake_model_interface. When including a new wake model in WAYVE, it
# suffices to implement the interface defined there.
#
# Finally, since some parametrizations, such as DispersiveStresses, require the wake model velocities to be evaluated on
# a dense grid, the VaryingBackground coupling methods can be instantiated with a WakeModelVelocityHandler object, which
# will perform the necessary calculations. However, since this is expensive, this should not be done unless necessary.
#
# In this example, we will use the VelocityMatching method with the wake merging method of Lanzilao and Meyers (2022).

# Here, we use the uni-directional wake merging method of Lanzilao and Meyers (2022). The default wake model settings
# are explained in Devesse et al. (2024).
wake_model = Lanzilao(wake_deflection=False)

# Since we use the velocity matching method and a parametrization for the dispersive stresses, we require a
# WakeModelVelocityHandler object.
subgrid_res = 8     # Ratio of turbine diameter and subgrid spacing
wm_velocity_handler = SelfSimilarWMVH(subgrid_res)

# Initialize the coupling object
shape_frac = 0.4    # Ratio of the filter length to the shape function spacing in each direction in the VM least-squares problem
coupling = VelocityMatching(wake_model, wm_velocity_handler, shape_frac)


# --------------------------------------------------- #
# -------- Step 3: set up wind farm object ---------- #
# --------------------------------------------------- #

# Gaussian filter length
Lfilter = 1000.

# Generate wind farm object
wind_farm = WindFarm(turbines, Lfilter, coupling)


# ------------------------------------------------------ #
# -------- Step 4: additional wind farm effects -------- #
# ------------------------------------------------------ #

# Dispersive stresses within the farm
disp_stresses = DispersiveStresses(wind_farm)

# Increased turbulent momentum flux above the farm
dmfp = ConstantFlux(wind_farm)

# Combined forcing object.
# The composite structure allows the APM to treat all these forcing terms as a single object. It's important to note
# that the elements of the composite are pre- and re-processed in order, so the wind farm should be placed before the
# other terms.
forcing = ForcingComposite([wind_farm,
                            disp_stresses,
                            dmfp])


# ---------------------------------------------------- #
# ---------------- Step 5: set up APM ---------------- #
# ---------------------------------------------------- #

# Use the atmospheric conditions from the example of Allaerts and Meyers (2019).
sub = True      # Sub- or supercritical case
abl = AM2019(sub)

# Generate 2D grid object
Nx = 2000           # grid points in x-direction
Lx = 1.e6           # grid size in x-direction [m]
Ny = 800            # grid points in y-direction
Ly = 0.4e6          # grid size in y-direction [m]
grid = Stat2Dgrid(Lx, Nx, Ly, Ny)

# Momentum flux parametrization
mfp = FrictionCoefficients()

# Pressure feedback parametrization
pressure = Uniform(dynamic=True, rotating=False)

# Create static 2D model
model = APM(grid, forcing, abl, mfp, pressure)


# ----------------------------------------------------- #
# --------------- Step 6: solve the APM --------------- #
# ----------------------------------------------------- #

# Since the wind farm forcing is non-linear, we use a fixed-point iteration solver. At each step, the forcing terms are
# re-processed, which includes the updating of the background velocities calculated by the coupling method, and
# re-running the wake model.

# Wse a fixed-point iteration solver with a relaxation factor of 0.7.
tol = 1.e-3     # Tolerance threshold
max_iter = 10   # Maximum number of iterations
relax = 0.7     # Relaxation factor
solver = FixedPointIteration(tol, max_iter, relax)

# Solve APM equations
result = model.solve(solver, verbose=True)


# ------------------------------------------------------ #
# ------------ Some example post processing ------------ #
# ------------------------------------------------------ #

# Evaluate wind-farm power output. The WindFarm class provides a calculation based on the coupling to the wake model.
turbine_powers = wind_farm.power_turbines(abl.rho)  # List of turbine power outputs
print(f"Total power production = {np.sum(turbine_powers)/1.0e9:.16f} GW")

# Layout plot #
fig, ax = plt.subplots()
im = ax.scatter(wind_farm.xs/1.e3, wind_farm.ys/1.e3,
                c=turbine_powers/1.e6,
                marker='o', edgecolors='k',
                rasterized=True)
cbar = fig.colorbar(im, ax=ax, shrink=0.6, label=r'Turbine power output [MW]')
ax.set_aspect('equal', 'box')
ax.set_xlabel(r'$x\;[\mathrm{km}]$')
ax.set_ylabel(r'$y\;[\mathrm{km}]$')
plt.show()

# Once the coupling object has been (p)re-processed, which happens during the fixed-point iteration, it provides
# functions to evaluate both the background velocity and the APM lower layer state in a region around the wind farm.
u_bg_evaluator = coupling.set_up_u_bg_evaluator(abl)    # Background velocity callable
apm_evaluator = coupling.apm_evaluator                  # APM lower layer state callable

# Hub height velocity plots #
# Region around wind farm, where the coupling calculations are supported
x_min, x_max, y_min, y_max = coupling.region_around_farm(wind_farm)
# Unperturbed velocity
s_0 = np.sqrt(abl.u(zh)**2 + abl.v(zh)**2)
# Set up xy grid at hub height
N_sg = 150
x = np.linspace(x_min, x_max, N_sg)
y = np.linspace(y_min, y_max, N_sg)
z = zh
# Get hub height velocities
u_bg, v_bg, u_wm, v_wm = wake_model.xy_plane(wind_farm, abl, u_bg_evaluator, apm_evaluator, x, y, z)
s_bg = np.sqrt(np.square(u_bg) + np.square(v_bg))
s_wm = np.sqrt(np.square(u_wm) + np.square(v_wm))
# Set up plot
f, axarr = plt.subplots(1, 2, figsize=(11., 5.))
f.subplots_adjust(wspace=0.5)
x_lims = [x_min/1.e3, x_max/1.e3]
y_lims = [y_min/1.e3, y_max/1.e3]
# Background velocity at hub height
im = axarr[0].pcolormesh(x / 1.e3, y / 1.e3, (s_bg - s_0).T / s_0,
                         shading='gouraud',
                         cmap='RdBu_r',
                         norm=CenteredNorm(),
                         rasterized=True)
f.colorbar(im, ax=axarr[0], shrink=0.6, label=r"$(U_{b}-U_0)/U_0$ [-]")
# Wake model velocity at hub height
im = axarr[1].pcolormesh(x / 1.e3, y / 1.e3, (s_wm - s_0).T / s_0,
                         shading='gouraud',
                         cmap='RdBu_r',
                         norm=CenteredNorm(),
                         rasterized=True)
f.colorbar(im, ax=axarr[1], shrink=0.6, label=r"$(u_{wm}-U_0)/U_0$ [-]")
# Add turbines, labels, and limits to both axes
for ax in axarr:
    # Plot turbines
    for i in range(Nt):
        ax.plot([wind_farm.turbines[i].x/1.e3, wind_farm.turbines[i].x/1.e3],
                [wind_farm.turbines[i].y/1.e3-D/2.e3, wind_farm.turbines[i].y/1.e3+D/2.e3], '-k')
    # Add axes labels and limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel(r'$x\;[\mathrm{km}]$')
    ax.set_ylabel(r'$y\;[\mathrm{km}]$')
# Show plot
plt.show()
