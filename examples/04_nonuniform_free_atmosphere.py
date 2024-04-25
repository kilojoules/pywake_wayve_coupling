#!/usr/bin/env python

'''
A script for an example case with a non-uniform upper atmosphere, based on the cases from Devesse et al. (2022).

References
----------
..  Devesse, K., Lanzilao, L., Jamaer, S., Van Lipzig, N., & Meyers, J. (2022). Including realistic upper atmospheres in
 a wind-farm gravity-wave model. Wind Energy Science, 7(4), 1367–1382. https://doi.org/10.5194/wes-7-1367-2022
..  Nieuwstadt, F. T. M. (1983). On the solution of the stationary, baroclinic Ekman-layer equations with a finite
 boundary-layer height. Boundary-Layer Meteorology, 26(4), 377–390. https://doi.org/10.1007/BF00119534
..  Allaerts, D., & Meyers, J. (2019). Sensitivity and feedback of wind-farm-induced gravity waves. Journal of Fluid
 Mechanics, 862, 990–1028. https://doi.org/10.1017/jfm.2018.969
..  Wells, H., & Vosper, S. B. (2010). The accuracy of linear theory for predicting mountain-wave drag: Implications for
 parametrization schemes. Quarterly Journal of the Royal Meteorological Society, 136(647), 429–441.
 https://doi.org/10.1002/qj.578
'''

__author__ = "Devesse Koen"
__date__ = "April 21, 2021"

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from wayve.apm import APM
from wayve.abl.abl_setup import analytic_cubic
from wayve.abl.abl_tools import Cg_cubic, alpha_cubic
from wayve.grid.grid import Stat2Dgrid
from wayve.forcing.apm_forcing import CST
from wayve.solvers import NoFeedback
from wayve.momentum_flux_parametrizations import FrictionCoefficients
from wayve.pressure.gravity_waves.gravity_waves import NonUniform
from wayve.pressure.gravity_waves.wave_patterns import get_du_field, get_dp_field
from examples.plotting import apm_plot, plot_field


# --------------------------------------------------- #
# ------------- Step 1: abl setup ------------------- #
# --------------------------------------------------- #

# We set up an ABL object based on the analytic solution for the velocity profiles in the ABL found by Nieuwstadt
# (1983). Since the resulting ABL does not include a free atmosphere, as only the profiles are only defined within the
# ABL, we then add a free atmosphere to the ABL object.
#
# All the parameters are documented in Devesse et al. (2022), and are based on combining the flow conditions used by
# Allaerts and Meyers (2019) with the idealized upper atmosphere studied by Wells and Vosper (2010).

# Input parameters
fc = 1.0e-4                                 # Coriolis parameter [1/s]
N = 0.0113                                  # Brunt Vaisala frequency [1/s]
G = 20.1                                    # Geostrophic wind speed [m/s]
kappa = 0.41                                # Von Karman constant [-]
H1 = 240.                                   # Wind-farm layer height (2*zh) [m]
dth = 6.076                                 # Inversion parameter   [K]
hstar = 0.15                                # Non-dimensional boundary layer height [-]
z0_h = 1.e-4                                # Non-dimensional surface roughness length [-]
Cg = Cg_cubic(hstar, z0_h, kappa)           # Geostrophic drag Cg = utau/G
alpha = alpha_cubic(hstar, z0_h, kappa)     # Geostrophic wind angle
utau = Cg * G                               # Friction velocity [m/s]
h = hstar * utau / fc                       # Boundary-layer height [m]
z0 = z0_h * h                               # Surface roughness
TI = 0.12                                   # Turbulence intensity

# ABL object setup
abl = analytic_cubic(dth, fc, N, G, alpha, kappa, utau, h, z0, H1, TI, 1000)

# Rotate abl so that wind is aligned in the free atmosphere
abl.rotate(abl.WD3*np.pi/180.0)

# Add free atmosphere of Wells and Vosper (2010) #

# Height-dependent upper atmosphere profiles
# N parameters
zn1 = 1300. + h
zn2 = 2000. + h
zn3 = 11000. + h
N0 = 0.01
N1 = 0.019
N2 = 0.011
dN2dz = -0.5e-9
N3 = 0.027
# N spline
N_spline = UnivariateSpline([h, zn1, zn2, zn3, zn3+1.e-8],
                            [N0, N1, N2, N2 + dN2dz * (zn3 - zn2), N3],
                            k=1, s=0, ext=3)
# U parameters
zu0 = 2900. + h
zu1 = 10000. + h
U0 = 14
U1 = 27
# U spline
U_spline = UnivariateSpline([h, zu0 / 2., zu0, (zu0 + zu1) / 2., zu1, zu1 + 500., zu1 + 750., zu1 + 1000.],
                            [U0, U0, U0, (U0 + U1) / 2., U1, U1, U1, U1],
                            k=1, s=0, ext=3)
# V spline
V_spline = UnivariateSpline([0., zu0 / 2., zu0],
                            [0., 0., 0.],
                            k=1, s=0, ext=3)
# Set up vertical grid
# dh = 12.5     # grid spacing used by Wells and Vosper (2010)
dh = 100.       # grid spacing that resolves faster (Note: this grid spacing is only used if n_layers <= 0 below)
h_upper = zn3 + dh
n_layers = int(np.rint((h_upper - h) / dh)) + 1
alt = np.linspace(h, h_upper, n_layers, endpoint=True)
alt[1:] -= dh / 2.
alt = np.append(alt, 15.e3)
# Set up input arrays
us = U_spline(alt)
vs = V_spline(alt)
Ns = N_spline(alt)
# Solve for theta profile
ths = (288.15 + dth) * np.ones(Ns.shape)
for i in range(len(ths)):
    ths[i] += 288.15 / 9.81 * np.trapz(np.power(Ns[:i+1], 2), alt[:i+1])

# Add FA to abl
abl.add_FA(alt, us, vs, Ns, ths,
           abl.U3, abl.V3, abl.N,
           h_upper, U1, 0., N3)

# Rotate abl so that wind is aligned in the wind-farm layer
abl.rotate(abl.WD1*np.pi/180.0)


# ---------------------------------------------------- #
# ------- Step 2: set pressure parametrization ------- #
# ---------------------------------------------------- #

# The vertical structure of the free atmosphere is included by using gravity wave theory with the piecewise method of
# Devesse et al. (2022). In this process, the free atmosphere is divided into layers, in which the vertical wavelength
# is taken as a constant value. This adds computational cost when calculating the stratification coefficients Phi, which
# is done once when setting up the solver routine.

# Piecewise method settings #
# Number of sublayers used by the piecewise method. The sublayers will be evenly spaced between the capping inversion
# and the tropopause.
# If n_layers <= 0, the vertical grid of the ABL (zs) is used to set up the sublayers. This is recommended when the
# vertical resolution of the ABL object profiles is low, since using more sublayers will then not necessarily result in
# a more accurate or realistic solution.
n_layers = 20
# The order of the splines used to determine the atmospheric variables when setting up the vertical wavelengths and
# sublayer interface conditions in the piecewise method.
spline_order = 1

# Set up non-uniform gravity wave object
pressure = NonUniform(n_layers=n_layers, order=spline_order)


# ---------------------------------------------------- #
# ---------------- Step 3: set up APM ---------------- #
# ---------------------------------------------------- #

# For simplicity, we use the same CST forcing object as the first example.
Ct = 0.006      # Drag coefficient
length = 20.e3  # Size of the forcing region in the x-direction [m]
width = 30.e3   # Size of the forcing region in the y-direction [m]
vertices = np.array([[-length/2, -width/2],
                     [length/2, -width/2],
                     [length/2, width/2],
                     [-length/2, width/2],
                     [-length/2, -width/2]])
forcing = CST(Ct, vertices)

# Generate 2D grid object
Nx = 2000           # grid points in x-direction
Lx = 1.e6           # grid size in x-direction [m]
Ny = 800            # grid points in y-direction
Ly = 0.4e6          # grid size in y-direction [m]
grid = Stat2Dgrid(Lx, Nx, Ly, Ny)

# Momentum flux parametrization
mfp = FrictionCoefficients()

# Create static 2D model
model = APM(grid, forcing, abl, mfp, pressure)


# ------------------------------------------------- #
# ------------- Step 3: solve the APM ------------- #
# ------------------------------------------------- #

# The additional cost from including variations in the upper atmosphere comes purely from calculating the stratification
# coefficients Phi when pre-processing the APM.
solver = NoFeedback()

# Solve APM equations
result = model.solve(method=solver, verbose=True)

# ------------------------------------------------- #
# ------------ Example post processing ------------ #
# ------------------------------------------------- #

# Difference in velocity magnitude in wind-farm layer
M1 = -(abl.S1-np.sqrt((abl.U1+result['u1r'])**2 +
                      (abl.V1+result['v1r'])**2)) / abl.S1*100
# Difference in velocity magnitude in upper layer
M2 = -(abl.S2-np.sqrt((abl.U2+result['u2r'])**2 +
                      (abl.V2+result['v2r'])**2)) / abl.S2*100

# Plot boundaries (in km)
lim = 100.
x_lims = [-lim, lim]
y_lims = [-lim, lim]

# Field plots #
# Set up plot
f, axarr = plt.subplots(1, 3, figsize=(15., 5.))  # Make sure there is enough room for the colorbars
f.subplots_adjust(wspace=0.75)
# Plot variables
plot_field(result['etar'] / abl.H * 100, grid, r'$\eta_t/H\;[\%]$', axarr[0], f, x_lims, y_lims)
plot_field(result['pr'] / abl.S1 ** 2 * 100, grid, r'$p^\prime/\rho_0U_1^2\;[\%]$', axarr[1], f, x_lims, y_lims)
plot_field(M1, grid, r'$u^\prime_1/U_1\;[\%]$', axarr[2], f, x_lims, y_lims)
# Plot outline of forcing term
for ax in axarr:
    ax.plot(vertices[:, 0] / 1.e3, vertices[:, 1] / 1.e3, '-k')
# Show plot
plt.show()

# APM plots #
# Calculate gravity wave fields in the free atmosphere
Nz_upper = 100
z_upper = np.linspace(abl.H, 15.e3, Nz_upper, endpoint=True)
print(f"Start calculations for gravity wave visualisation")
u_wave = get_du_field(result['etac'], z_upper, grid, abl, pressure) / abl.S3 * 100.
p_wave = get_dp_field(result['etac'], z_upper, grid, abl, pressure) / abl.S3**2*100.
print(f"Gravity wave calculations done")
# Combine variables into input lists
etas = [result['eta1r'], result['eta2r']]
u_fields = [M1*abl.S1/abl.S3, M2*abl.S2/abl.S3]
p_fields = [result['p1r']/abl.S3**2*100, result['p2r']/abl.S3**2*100]
# Velocity perturbation #
# Set up plot
f, ax = plt.subplots()
# Perform APM plot
im = apm_plot(u_fields, grid, abl, etas, u_wave, z_upper, ax, x_lims, forcing=forcing)
# Add colorbar
f.colorbar(im, ax=ax, shrink=1.0, label=r"$\Delta u/U_g$ [%]")
# Show plot
plt.show(block=False)
# Pressure perturbation #
# Set up plot
f, ax = plt.subplots()
# Perform APM plot
im = apm_plot(p_fields, grid, abl, etas, p_wave, z_upper, ax, x_lims, forcing=forcing)
# Add colorbar
f.colorbar(im, ax=ax, shrink=1.0, label=r'$p/\rho_0U_g^2\;[\%]$')
# Show plot
plt.show()
