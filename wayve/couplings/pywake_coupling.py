
import numpy as np
from wayve.forcing.wind_farms.wake_model_coupling.wake_model_interface import UniDirectionalSelfSimilar
import py_wake
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.wind_farm_models import PropagateDownwind, All2AllIterative
from py_wake.turbulence_models import STF2005TurbulenceModel, STF2017TurbulenceModel, CrespoHernandez
from py_wake.rotor_avg_models import RotorCenter, GridRotorAvg
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine
from py_wake.site.xrsite import XRSite
import xarray as xr


class PyWakeInterface(UniDirectionalSelfSimilar):

    def __init__(self, k=0.04, deficit_model=BastankhahGaussianDeficit, rotor_avg_model=RotorCenter, 
                 superposition_model=, turbulence_mode=CrespoHernandez, blockage_model=None,
                 wind_farm_model=PropagateDownwind, deficit_kwargs=None):

        # TODO: Needs more setup options (eg. wake model settings, turbulence model used, ...)
        self.deficit_model = deficit_model
        self.rotor_avg_model = rotor_avg_model
        self.superposition_model = superposition_model
        self.turbulence_mode = turbulence_mode
        self.blockage_model = blockage_model
        self.wind_farm_model = wind_farm_model
        self.deficit_kwargs = deficit_kwargs
        self.__k = k

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, value):
        self.__k = value

    def set_up_pywake_wind_farm(self, wind_farm, abl, u_bg_evaluator, apm_evaluator):
        # Turbine hub locations
        Nt = wind_farm.Nturb
        xs = wind_farm.xs
        ys = wind_farm.ys
        hhs = [wind_farm.turbines[ii].zh for ii in range(Nt)]
        locs = np.column_stack([xs, ys, hhs])
        # Determine unperturbed background wind
        hh = hhs[0]     # assume the same hub heights for now
        U0 = abl.u(hh)
        V0 = abl.v(hh)
        # Wind speed
        inflow_speed = np.sqrt(U0 ** 2 + V0 ** 2)
        # Determine the main wind direction
        e_str, e_span = self.background_flow_direction(wind_farm, abl)  # unit vector in main and spanwise wind directions
        wd = np.arctan2(e_str[1], e_str[0])     # Wind direction in radians
        # Determine background flow
        background_flow = u_bg_evaluator(locs)
        background_flow_str = background_flow[:, 0] * e_str[0] + background_flow[:, 1] * e_str[1]   # Project on wd
        # PyWake call
        dummy_speeds = np.linspace(1, 30)
        site_ds = xr.Dataset(
            data_vars={
            'Speedup': ('i', background_flow_str / inflow_speed),
            'P': 1,
            'TI': abl.TI*100,
            #'P': ('ws', np.ones_like(dummy_speeds) / dummy_speeds.size)
            },
            coords={
                'i': range(Nt),
            'ws': dummy_speeds})
        site = XRSite(site_ds, initial_position=np.array([xs, ys]).T)

        # assuming a single turbine type for now.
        u = np.linspace(0, 30)
        cp = wind_farm.turbines[0].Cp(u)
        D = wind_farm.turbines[0].D
        power = 1.225 * cp * D ** 2 * np.pi / 4
        ct = wind_farm.turbines[0].Ct(u)
        turbine = WindTurbine(name='wayve-turbine',
                              diameter=D,
                              hub_height=hh,
                              powerCtFunction=PowerCtTabular(u, power, 'kW', ct))

        if self.deficit_kwargs is not None:
           d_kwargs = {'k': self.k}
        else:
           d_kwargs = self.deficit_kwargs

        if self.turbulenceModel is None:
           turb_model = None
        else:
           turb_model = self.turbulenceModel()

        # note that blockage only makes sense with All2All wind farm model
        if self.blockage_model is None
           blockage = None
        else:
           blockage = self.blockage_model()

        self.windFarmModel = self.wind_farm_model(
                           site,
                           turbine,
                           wake_deficitModel=self.wake_deficitModel(**d_kwargs),
                           superpositionModel=self.superpositionModel()
                           deflectionModel=self.deflectionModel()
                           turbulenceModel=turb_model,
                           blockageModel=blockage)

        sim_res = self.windFarmModel(xs, ys, ws=inflow_speed, wd=270. - np.rad2deg(wd), TI=abl.TI, yaw=0, tilt=0)
        return (sim_res)
    def get_St_Ct_et(self, wind_farm, abl, u_bg_evaluator, apm_evaluator):

        # set up wind farm
        sim_res = self.set_up_pywake_wind_farm(self, wind_farm, abl, u_bg_evaluator, apm_evaluator)

        effective_speeds = sim_res.WS_eff.to_numpy()[:, 0, 0]
        thrust_coefficients = np.zeros(Nt)
        ets = np.zeros((Nt, 2))
        for i in range(Nt):
            thrust_coefficients[i] = wind_farm.turbines[i].Ct(effective_speeds[i])
            ets[i] = np.array([np.cos(wd), np.sin(wd)])
        return effective_speeds, thrust_coefficients, ets

    def get_u_subgrid(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculate the wake model velocity field on the given subgrid.

        This method will only be called after the method get_St_Ct_et has been called, and will use the same inputs.
        Therefore, feel free to store any intermediate results that can be re-used in this calculation.

        The basic turbine information can be found in the WindFarm object. The unperturbed atmospheric state can be
        found in the ABL object. Additionally, the u_bg_evaluator is a callable that can evaluate the background
        velocities at requested locations. Finally, the apm_evaluator is a callable that can evaluate the lower-layer
        APM wind speeds and height at requested locations.

        Parameters
        ----------
        wind_farm    WindFarm object
            Wind farm for which the calculation is performed
        abl     ABL object
            Information on the unperturbed background state, and atmospheric variables such as TI and z0.
        u_bg_evaluator  callable
            Callable which can evaluate the background velocities at requested locations.
            See VaryingBackground.set_up_u_bg_evaluator in varying_background.py for detailed documentation.
        apm_evaluator   callable
            Callable which can evaluate the layer wind speeds and height at requested locations.
            See VaryingBackground.set_up_apm_evaluators in varying_background.py for detailed documentation.
        subgrid     SubGrid object
            Grid on which the velocity should be evaluated

        Returns
        -------
        u_wm  np.array
            X-components of the wake model velocity field, defined on the subgrid (shape same as subgrid.shape)
        v_wm  np.array
            Y-components of the wake model velocity field, defined on the subgrid (shape same as subgrid.shape)
        """
        sim_res = self.set_up_pywake_wind_farm(self, wind_farm, abl, u_bg_evaluator, apm_evaluator)
        # TODO: Implement this method
        raise NotImplementedError("Velocity calculation not implemented yet!")

    def wake_deficit(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculate the wake deficit, calculated on the given subgrid.

        The wake deficit is the factor that, when multiplied with the background velocity, gives the wake model
        velocity. For the uni-directional wake merging method of Lanzilao and Meyers (Wind Energy, 2022), this
        corresponds to the product over all wake functions (1-W_k).
        """
        # Get main streamwise direction #
        e_str, e_span = self.background_flow_direction(wind_farm, abl)
        # Get wake model velocities #
        # XY components
        u_wm, v_wm = self.get_u_subgrid(wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid)
        # Map onto streamwise direction
        str_wm = u_wm * e_str[0] + v_wm * e_str[1]
        # Get background velocities #
        # SubGrid coordinates
        locations = subgrid.locations
        # Evaluate background velocities
        vel_bg = u_bg_evaluator(locations)
        # Reshape into grid
        Nx, Ny, Nz = subgrid.shape
        u_bg = np.reshape(vel_bg[:, 0], (Nx, Ny, Nz))
        v_bg = np.reshape(vel_bg[:, 1], (Nx, Ny, Nz))
        # Map onto streamwise direction
        str_bg = u_bg * e_str[0] + v_bg * e_str[1]
        # Calculate wake deficit #
        w = np.divide(str_wm, str_bg)
        return w

    def xy_plane(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, xs, ys, z):
        """
        Compute an xy cross-section of the flow.
        """
        # TODO: Implement this method
        raise NotImplementedError("Velocity calculation not implemented yet!")


