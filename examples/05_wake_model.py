#!/usr/bin/env python

"""
A basic example script that demonstrates how to run the underlying wake model, without the APM.

The WAYVE code was not designed to run wake models on their own. However, due to the modularity of the design, it is
possible to do so. In essence, we simply have the wind farm perform a pre-processing calculation, which includes a wake
model call, and don't do any additional operations.

In order to re-use the WAYVE framework, it is still required to set up several objects that do not influence the wake
model at all, but are expected by the code. These are, among others:
    - grid: a Stat2Dgrid object
    - Lfilter: a filter length for the Gaussian filtering
    - mfp: a momentum flux parametrization
    - pressure: a pressure closure equation
While these objects do not affect the wake model output, users are still encouraged to set up meaningful default inputs
for them, as is done in this script. Otherwise, the framework might crash in unexpected places.

References
----------
..  Allaerts, D., & Meyers, J. (2019). Sensitivity and feedback of wind-farm-induced gravity waves. Journal of Fluid
 Mechanics, 862, 990–1028. https://doi.org/10.1017/jfm.2018.969
..  Lanzilao, L., & Meyers, J. (2024). A parametric large-eddy simulation study of wind-farm blockage and gravity waves
 in conventionally neutral boundary layers. Journal of Fluid Mechanics, 979, A54. https://doi.org/10.1017/jfm.2023.1088
..  Lanzilao, L., & Meyers, J. (2022). A new wake‐merging method for wind‐farm power prediction in the presence of
 heterogeneous background velocity fields. Wind Energy, 25(2), 237–259. https://doi.org/10.1002/we.2669
..  Devesse, K., Lanzilao, L., & Meyers, J. (2023). A meso-micro atmospheric perturbation model for wind farm blockage.
 Preprint. http://arxiv.org/abs/2310.18748
"""

__author__ = "Koen Devesse"
__date__ = "October 4, 2023"

import numpy as np
import matplotlib.pyplot as plt

from wayve.apm import APM
from wayve.abl.abl_setup import AM2019
from wayve.grid.grid import Stat2Dgrid
from wayve.forcing.wind_farms.wake_model_coupling.wake_models.lanzilao_merging import UniDirectional
from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import PureWM
from wayve.forcing.wind_farms.wind_farm import WindFarm, Turbine
from wayve.momentum_flux_parametrizations import FrictionCoefficients
from wayve.pressure.pressure_parametrizations import NoPressureFeedback


# ----------------------------------------------------- #
# ------------- Step 1: set up turbines --------------- #
# ----------------------------------------------------- #

# We use the same turbines and farm layout as Lanzilao and Meyers (2024), which is also used in 02_wind_farm.py.
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

# List of Turbine objects
turbines = []
for t in range(Nt):
    turbine = Turbine(X[t], Y[t], D, zh, Ct)
    turbines.append(turbine)


# ------------------------------------------------------ #
# ------- Step 2: set up wake model and coupling ------- #
# ------------------------------------------------------ #

# Here, we use the uni-directional wake merging method of Lanzilao and Meyers (2022). The default wake model settings
# are explained in Devesse et al. (2023).
wake_model = UniDirectional()

# Set up coupling object. We use a PureWM object, which inherits from VaryingBackground, and always assumes the
# background velocity is unchanged.
coupling = PureWM(wake_model)

# Note that we avoid initializing a WakeModelVelocityHandler object, as this would unnecessarily increase the
# computational cost.


# ---------------------------------------------------- #
# ------------ Step 3: define wind farm -------------- #
# ---------------------------------------------------- #

# Gaussian filter length (meaningless for uncoupled wake model run)
Lfilter = 1000.

# Generate wind farm object
wind_farm = WindFarm(turbines, Lfilter, coupling)


# ---------------------------------------------------- #
# ---------------- Step 4: set up APM ---------------- #
# ---------------------------------------------------- #

# Use the atmospheric conditions from the example of Allaerts and Meyers (2019).
sub = True      # Sub- or supercritical case
abl = AM2019(sub)

# Generate 2D grid object (just has to be big enough to cover the farm)
Nx = 80     # grid points in x-direction
Lx = 1.e5   # grid size in x-direction [m]
Ny = 80     # grid points in y-direction
Ly = 1.e5   # grid size in y-direction [m]
grid = Stat2Dgrid(Lx, Nx, Ly, Ny)

# APM components, meaningless in this context
mfp = FrictionCoefficients()
pressure = NoPressureFeedback()
model = APM(grid, wind_farm, abl, mfp, pressure)


# ------------------------------------------------------ #
# --------------- Step 5: Run wake model --------------- #
# ------------------------------------------------------ #

# Pre-processing the wind farm object includes an initial run of the wake model, assuming unperturbed background
# velocity. There will be some additional calculations, but these are relatively fast.
# Note that in a script like this, there are also relatively high costs associated with the initial Numba compilations.
print("Start wake model run")
wind_farm.preprocess(model)
print("Wake model run done")

# Evaluate turbine power outputs
powers = wind_farm.power_turbines(abl.rho)


# ------------------------------------------------------ #
# ------------ Some example post processing ------------ #
# ------------------------------------------------------ #

# Layout plot #
fig, ax = plt.subplots()
im = ax.scatter(wind_farm.xs/1.e3, wind_farm.ys/1.e3,
                c=powers/1.e6,
                marker='o', edgecolors='k',
                rasterized=True)
cbar = fig.colorbar(im, ax=ax, shrink=0.7, label=r'Turbine power output [MW]')
ax.set_aspect('equal', 'box')
ax.set_xlabel(r'$x\;[\mathrm{km}]$')
ax.set_ylabel(r'$y\;[\mathrm{km}]$')
plt.show()

# Hub height velocity plot #
print("Start hub height velocity calculation")
# Region around wind farm
x_min, x_max, y_min, y_max = coupling.region_around_farm(wind_farm)
# Unperturbed velocity
s_0 = np.sqrt(abl.u(zh)**2 + abl.v(zh)**2)
# Set up xy grid at hub height
N_sg = 150
x = np.linspace(x_min, x_max, N_sg)
y = np.linspace(y_min, y_max, N_sg)
z = zh
# Callables to evaluate the background velocity and the APM lower layer state
u_bg_evaluator = coupling.set_up_u_bg_evaluator(abl)
apm_evaluator = coupling.apm_evaluator
# Get hub height velocities
_, _, u_wm, v_wm = wake_model.xy_plane(wind_farm, abl, u_bg_evaluator, apm_evaluator, x, y, z)
s_wm = np.sqrt(np.square(u_wm) + np.square(v_wm))
print("Hub height velocity calculation done")
# Set up plot
f, ax = plt.subplots()
x_lims = [x_min/1.e3, x_max/1.e3]
y_lims = [y_min/1.e3, y_max/1.e3]
# Wake model velocity at hub height
im = ax.pcolormesh(x / 1.e3, y / 1.e3, s_wm.T / s_0,
                   shading='gouraud',
                   rasterized=True)
f.colorbar(im, ax=ax, shrink=0.7, label=r"$u_{wm}/U_0$ [-]")
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
