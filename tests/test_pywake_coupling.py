# tests/test_pywake_coupling.py

import pytest
import numpy as np

# WAYVE imports
from wayve.abl import abl_setup
from wayve.forcing.wind_farms.wind_farm import WindFarm, Turbine
from wayve.couplings.pywake_coupling import PyWakeInterface
from wayve.forcing.wind_farms.wake_model_coupling.coupling_methods.varying_background import SubGrid

# PyWake imports
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.superposition_models import SquaredSum
from py_wake.flow_map import Points
from foxes.utils import uv2wd


@pytest.fixture(scope="module")
def abl():
    """Provides a standard, pre-configured ABL object for tests."""
    return abl_setup.AM2019(sub=True, Nz=100)



@pytest.fixture(scope="module")
def wind_farm_side_by_side():
    """Provides a simple two-turbine wind farm with a side-by-side layout."""
    
    # Define a simple Ct curve that varies with wind speed for the test
    # This will produce different Ct values for different inflow speeds.
    ct_curve = lambda u: 0.9 - 0.01 * np.asarray(u)

    # Define two identical turbines placed apart in the crosswind direction
    turbine1 = Turbine(xloc=0., yloc=-500., diameter=126., zh=90.,
                       ct=ct_curve)
    turbine2 = Turbine(xloc=0., yloc=500., diameter=126., zh=90.,
                       ct=ct_curve)
                       
    return WindFarm(turbines=[turbine1, turbine2], Lfilter=200., coupling=None)




@pytest.fixture
def u_bg_evaluator_heterogeneous_side_by_side():
    """
    Creates a background velocity evaluator that assigns a different
    wind speed to each of the two side-by-side turbines.
    """
    u_bg_low_ws = np.array([8.0, 0.0])  # Corresponds to 270 deg WD
    u_bg_high_ws = np.array([12.0, 0.0]) # Corresponds to 270 deg WD

    def evaluator(locations):
        # locations is an array of shape (n_points, 3)
        velocities = np.zeros((locations.shape[0], 2))
        
        # Assign low speed to turbine at y=-500 and high speed to turbine at y=500
        is_turb0 = np.isclose(locations[:, 1], -500)
        is_turb1 = np.isclose(locations[:, 1], 500)
        
        velocities[is_turb0, :] = u_bg_low_ws
        velocities[is_turb1, :] = u_bg_high_ws
        
        # Handle other points (e.g., for subgrid)
        # Assign velocity based on which turbine is closer
        is_other = ~is_turb0 & ~is_turb1
        if np.any(is_other):
            velocities[is_other & (locations[:, 1] < 0)] = u_bg_low_ws
            velocities[is_other & (locations[:, 1] >= 0)] = u_bg_high_ws
        
        return velocities
        
    return evaluator


@pytest.fixture
def pywake_coupling():
    """Provides a standard PyWakeInterface coupling object."""
    return PyWakeInterface(
        deficit_model=BastankhahGaussianDeficit,
        superposition_model=SquaredSum()
    )


def test_get_st_ct_et_no_wake_interaction(wind_farm_side_by_side, abl, u_bg_evaluator_heterogeneous_side_by_side, pywake_coupling):
    """
    Tests that with a side-by-side layout, each turbine's inflow velocity (St)
    matches its specified background velocity, as there is no wake interaction.
    """
    # Arrange
    wind_farm = wind_farm_side_by_side
    u_bg_evaluator = u_bg_evaluator_heterogeneous_side_by_side
    apm_evaluator = lambda x, y: (None, None, None) # No APM feedback

    # Act
    St, Ct, et = pywake_coupling.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, apm_evaluator)

    # Assert
    # Inflow speeds should match the heterogeneous background since there's no wake
    np.testing.assert_allclose(St[0], 8.0, rtol=1e-6)
    np.testing.assert_allclose(St[1], 12.0, rtol=1e-6)
    
    # Thrust coefficients should differ because inflow speeds differ
    assert not np.isclose(Ct[0], Ct[1])


def test_wake_deficit_from_isolated_turbines(wind_farm_side_by_side, abl, u_bg_evaluator_heterogeneous_side_by_side, pywake_coupling):
    """
    Tests if the wake deficit from two isolated turbines correctly reflects their
    different inflow speeds.
    """
    # Arrange
    wind_farm = wind_farm_side_by_side
    u_bg_evaluator = u_bg_evaluator_heterogeneous_side_by_side
    apm_evaluator = lambda x, y: (None, None, None) # No APM feedback
    
    # Run get_St_Ct_et first to set up the internal state of the coupling
    pywake_coupling.get_St_Ct_et(wind_farm, abl, u_bg_evaluator, apm_evaluator)

    # Create a subgrid with two points: one directly downstream of each turbine
    downstream_dist = 5 * wind_farm.turbines[0].D
    point0_coords = [downstream_dist, -500, wind_farm.turbines[0].zh]
    point1_coords = [downstream_dist, 500, wind_farm.turbines[1].zh]
    
    subgrid = SubGrid(Nx=2, Ny=1, Nz=1,
                      x_min=downstream_dist, x_max=downstream_dist,
                      y_min=-500, y_max=500,
                      z_min=wind_farm.turbines[0].zh, z_max=wind_farm.turbines[0].zh)
    # Correct the locations to match our specific points
    subgrid._SubGrid__locations = np.array([point0_coords, point1_coords])


    # Act
    wake_deficit_field = pywake_coupling.wake_deficit(wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid)
    
    # Reshape if necessary and extract values
    wake_deficit = wake_deficit_field.flatten()
    deficit_behind_turb0 = wake_deficit[0]
    deficit_behind_turb1 = wake_deficit[1]
    
    # Assert
    # The wake deficit is (U_bg - U_wake) / U_bg. A stronger wake means a larger deficit.
    # Turbine 1 has a higher inflow speed (12m/s vs 8m/s), so it should produce a stronger wake.
    assert deficit_behind_turb1 > deficit_behind_turb0
    assert deficit_behind_turb0 > 0.01 # Sanity check that there is some deficit
