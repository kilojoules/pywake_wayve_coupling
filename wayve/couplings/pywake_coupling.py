import numpy as np
from wayve.forcing.wind_farms.wake_model_coupling.wake_model_interface import UniDirectionalSelfSimilar
import py_wake
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.wind_turbines import WindTurbines
from py_wake.site import UniformSite
from foxes.utils import uv2wd
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.flow_map import HorizontalGrid
from py_wake.flow_map import Points 
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.turbulence_models import STF2005TurbulenceModel, STF2017TurbulenceModel, CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter, GridRotorAvg
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
from py_wake.site.xrsite import XRSite
import xarray as xr

class PyWakeInterface(UniDirectionalSelfSimilar):

    def __init__(self, deficit_model=BastankhahGaussianDeficit, rotor_avg_model=RotorCenter(),
                 superposition_model=LinearSum(), turbulence_model=CrespoHernandez(), deflection_model=JimenezWakeDeflection(),
                 blockage_model=None, wind_farm_model=PropagateDownwind, deficit_kwargs={}):

        self.deficit_model = deficit_model
        self.rotor_avg_model = rotor_avg_model
        self.superposition_model = superposition_model
        self.turbulence_model = turbulence_model
        self.deflection_model = deflection_model
        self.blockage_model = blockage_model
        self.wind_farm_model = wind_farm_model
        self.deficit_kwargs = deficit_kwargs
        self.sim_res = None


    def _run_simulation(self, wind_farm, abl, u_bg_evaluator):

        pywake_turbines = WindTurbines(
            names=[f"Turbine_{i}" for i in range(wind_farm.Nturb)],
            diameters=[t.D for t in wind_farm.turbines],
            hub_heights=[t.zh for t in wind_farm.turbines],
            powerCtFunctions=[
                PowerCtTabular(
                    ws=np.arange(0, 31),
                    power=t.Cp(np.arange(0, 31)) * 0.5 * abl.rho * t.rotorarea * np.arange(0, 31)**3,
                    power_unit='W',
                    ct=t.Ct(np.arange(0, 31)))
                for t in wind_farm.turbines
            ]
        )

        turbine_coords = np.array([[t.x, t.y, t.zh] for t in wind_farm.turbines])
        bg_vels_at_turbines = u_bg_evaluator(turbine_coords)
        mean_wd = np.mean(uv2wd(bg_vels_at_turbines))
        
        # Use the mean wind speed as the reference for the simulation case
        ref_ws = np.mean(np.linalg.norm(bg_vels_at_turbines, axis=1))

        # 1. Define a spatial grid that is slightly larger than the required flow map region
        x_min, x_max, y_min, y_max = wind_farm.coupling.region_around_farm(wind_farm)
        buffer = 1.0 # meters
        x_coords = np.linspace(x_min - buffer, x_max + buffer, 50)
        y_coords = np.linspace(y_min - buffer, y_max + buffer, 50)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        grid_points = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, wind_farm.turbines[0].zh)], axis=-1)

        # 2. Evaluate background velocities and calculate speed-up factors relative to the reference WS
        bg_vels_grid = u_bg_evaluator(grid_points)
        inflow_speeds_xy = np.linalg.norm(bg_vels_grid, axis=1).reshape(xx.shape)
        speedup_xy = inflow_speeds_xy / ref_ws
        
        # 3. Create the XRSite using the 'Speedup' data variable
        ds = xr.Dataset(
            data_vars={
                'Speedup': (('x', 'y'), speedup_xy),
                'P': (('wd',), [1.0]),
                'TI': abl.TI
            },
            coords={
                'x': x_coords,
                'y': y_coords,
                'wd': [mean_wd]
            }
        )
        site = XRSite(ds, interp_method='nearest') # bounds='limit' is no longer needed due to buffer
        
        wfm_kwargs = {
            'site': site, 'windTurbines': pywake_turbines,
            'wake_deficitModel': self.deficit_model(**self.deficit_kwargs),
            'superpositionModel': self.superposition_model,
            'deflectionModel': self.deflection_model,
            'turbulenceModel': self.turbulence_model
        }
        
        if self.blockage_model and self.wind_farm_model is not py_wake.wind_farm_models.PropagateDownwind:
            wfm_kwargs['blockage_model'] = self.blockage_model
        
        wfm_instance = self.wind_farm_model(**wfm_kwargs)
        
        # Run the simulation for the single case defined by the reference wind speed and direction.
        # PyWake will internally multiply the reference WS by the spatial Speedup field from the site.
        self.sim_res = wfm_instance(
            x=wind_farm.xs, y=wind_farm.ys, h=[t.zh for t in wind_farm.turbines],
            type=np.arange(wind_farm.Nturb),
            wd=[mean_wd],
            ws=[ref_ws],
            yaw=0, tilt=0
        )

    def get_St_Ct_et(self, wind_farm, abl, u_bg_evaluator, apm_evaluator):
        self._run_simulation(wind_farm, abl, u_bg_evaluator)
        
        effective_speeds = self.sim_res.WS_eff.to_numpy().flatten()
        thrust_coefficients = self.sim_res.CT.to_numpy().flatten()
        
        e_str, _ = self.background_flow_direction(wind_farm, abl)
        ets = np.tile(e_str, (wind_farm.Nturb, 1))

        return effective_speeds, thrust_coefficients, ets

    def wake_deficit(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculate the wake deficit, calculated on the given subgrid.
    
        The wake deficit is the factor that, when multiplied with the background velocity, gives the wake model
        velocity.
        """
        # Ensure the simulation has been run
        self._run_simulation(wind_farm, abl, u_bg_evaluator)
    
        # Convert WAYVE's subgrid to a PyWake-compatible grid
        locs = subgrid.locations
        pywake_grid = Points(x=locs[:, 0], y=locs[:, 1], h=locs[:, 2])
    
        # Get the flow map for all points in the subgrid
        flow_map = self.sim_res.flow_map(grid=pywake_grid)
    
        # The wake deficit is the ratio of effective wind speed to ambient wind speed
        wake_deficit = flow_map.WS_eff.values / flow_map.WS.values
    
        # Reshape the result to match the original subgrid shape
        wake_deficit = wake_deficit.reshape(subgrid.shape)
    
        return wake_deficit


    def get_u_subgrid(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculates the full velocity field on the subgrid. This is now called
        directly by the generic WakeModelVelocityHandler.
        """
        self._run_simulation(wind_farm, abl, u_bg_evaluator)

        # Get the flow map for all points in the subgrid
        flow_map = self.sim_res.flow_map(grid=subgrid)
        
        # Get wind speed and direction, then convert to u, v components
        ws_grid = flow_map.WS_eff.values
        wd_grid = flow_map.WD.values
        wd_rad = np.deg2rad(270.0 - wd_grid)
        
        u_wm = ws_grid * np.cos(wd_rad)
        v_wm = ws_grid * np.sin(wd_rad)

        return u_wm, v_wm

    def xy_plane(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, xs, ys, z):
        """This method also reuses the stored simulation for 2D visualization slices."""
        self._run_simulation(wind_farm, abl, u_bg_evaluator)

        grid = HorizontalGrid(x=xs, y=ys, h=z)
        flow_map = self.sim_res.flow_map(grid=grid)

        # Ambient (background) velocities
        amb_ws = flow_map.WS.values.squeeze()
        amb_wd = flow_map.WD.values.squeeze()
        amb_wd_rad = np.deg2rad(270.0 - amb_wd)
        amb_u = (amb_ws * np.cos(amb_wd_rad))
        amb_v = (amb_ws * np.sin(amb_wd_rad))

        # Waked velocities
        ws = flow_map.WS_eff.values.squeeze()
        wd = flow_map.WD.values.squeeze()
        wd_rad = np.deg2rad(270.0 - wd)
        u_wm = (ws * np.cos(wd_rad))
        v_wm = (ws * np.sin(wd_rad))

        return amb_u.T, amb_v.T, u_wm.T, v_wm.T
