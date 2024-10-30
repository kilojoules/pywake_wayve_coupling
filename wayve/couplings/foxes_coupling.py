import numpy as np

from wayve.forcing.wind_farms.wake_model_coupling.wake_model_interface import UniDirectionalSelfSimilar

from foxes import Engine, get_engine, reset_engine
from foxes import WindFarm, Turbine, ModelBook
from foxes.core import States, TurbineType
from foxes.utils import uv2wd, wd2uv
from foxes.algorithms import Downwind
import foxes.variables as FV
import foxes.constants as FC

class WayveTurbineType(TurbineType):
    
    def __init__(self, ctc, cpc, H, D, **kwargs):
        super().__init__(D=D, H=H, **kwargs)
        self._ctc = ctc
        self._cpc = cpc
    
    def needs_rews2(self):
        return False
    
    def needs_rews3(self):
        return False

    def output_farm_vars(self, algo):
        return [FV.CT, FV.P]
    
    def calculate(self, algo, mdata, fdata, st_sel):
        
        ws = fdata[FV.REWS][st_sel]
        rho = fdata[FV.RHO][st_sel]
        
        fdata[FV.CT][st_sel] = self._ctc(ws)
        
        A = np.pi*(self.D/2)**2
        cp = self._cpc(ws)
        rho = fdata[FV.RHO][st_sel]
        fdata[FV.P][st_sel] = 0.5*cp*rho*A*ws**3
        
        return {FV.CT: fdata[FV.CT], FV.P: fdata[FV.P]}
    
class WayveStates(States):
    
    def __init__(self):
        super().__init__()
        self.u_bg_evaluator = None
        self.abl = None
    
    def size(self):
        return 1
    
    def weights(self, algo):
        return np.ones((1, algo.n_turbines), dtype=FC.DTYPE)
    
    def calculate(self, algo, mdata, fdata, tdata):
        # prepare:
        n_states = tdata.n_states
        assert n_states == 1
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        n_points = n_targets * n_tpoints
        points = tdata[FC.TARGETS][0].reshape(n_points, 3)
        
        # compute background:
        uv = self.u_bg_evaluator(points)
        ws = np.linalg.norm(uv, axis=-1)
        wd = uv2wd(uv, axis=-1)
        
        return {
            FV.WS: ws.reshape(n_states, n_targets, n_tpoints),
            FV.WD: wd.reshape(n_states, n_targets, n_tpoints),
            FV.TI: np.full((n_states, n_targets, n_tpoints), self.abl.TI),
            FV.RHO: np.full((n_states, n_targets, n_tpoints), self.abl.rho),
        }
        
class FoxesWakeModel(UniDirectionalSelfSimilar):
    
    def __init__(self, turbine_models=[], mbook=None, verbosity=1, **algo_pars):
        """
        Constructor.

        Parameters
        ----------
        turbine_models: dict or list
            Turbine models additional to the turbine type mmodel
            If list, the list of turbine model names for all turbines.
            If dict, the mapping from turbine index to list of turbine model names.
        mbook: foxes.ModelBook, optional
            The model book object
        verbosity: int
            The verbosity level, 0=silent
        algo_pars: dict, optional
            Arguments for the foxes.algorithms.Downwind constructor
            
        """
        super().__init__()
        self._mbook = ModelBook() if mbook is None else mbook
        self._tmodels = turbine_models
        self._verbosity = verbosity
        self._algo_pars = algo_pars
        self._farm = None
        self._states = None
        self._algo = None
        self._farm_results = None
        self._ttypes = {}
    
    def _get_turbine_type(self, wayve_farm, turbine_index, **kwargs):
        ctc = wayve_farm.turbines[turbine_index].Ct
        cpc = wayve_farm.turbines[turbine_index].Cp
        H = wayve_farm.turbines[turbine_index].zh
        D = wayve_farm.turbines[turbine_index].D
        
        wsk = np.arange(0,30,0.05)
        key = (tuple(ctc(wsk)), tuple(cpc(wsk)), H, D)
        if key not in self._ttypes:
            name = f"WayveTurbineType{len(self._ttypes)}"
            self._ttypes[key] = name
            self._mbook.turbine_types[name] = WayveTurbineType(
                ctc, cpc, H, D, name=name, **kwargs
            )
        
        return self._ttypes[key]
    
    def _setup_foxes(self, wind_farm, u_bg_evaluator, abl):
        
        if self._farm is None:
            self._farm = WindFarm()
            for ti, turbine in enumerate(wind_farm.turbines):
                if isinstance(self._tmodels, dict):
                    tmodels = self._tmodels[ti]
                else:
                    tmodels = self._tmodels
                ttype = self._get_turbine_type(wind_farm, ti)
                self._farm.add_turbine(Turbine(
                        xy=np.array([turbine.x, turbine.y]),
                        D=turbine.D,
                        H=turbine.zh,
                        turbine_models=[ttype] + tmodels,
                    ),
                    verbosity=self._verbosity,                       
                )
                
            self._states = WayveStates()

            self._algo = Downwind(
                farm=self._farm,
                states=self._states,
                mbook=self._mbook,
                verbosity=self._verbosity,
                **self._algo_pars,
            )

        self._states.u_bg_evaluator = u_bg_evaluator
        self._states.abl = abl
        
    def get_St_Ct_et(self, wind_farm, abl, u_bg_evaluator, apm_evaluator):
        """
        Calculate the turbine inflow velocities (St), the thrust coefficients (Ct), and the turbine directions (et).

        This method will always be called before the method get_St_Ct_et has been called, and will use the same inputs.
        Therefore, feel free to store any intermediate results that can be re-used.

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
        """
        self._setup_foxes(wind_farm, u_bg_evaluator, abl)
        
        self._farm_results = self._algo.calc_farm()
        
        St = self._farm_results[FV.REWS].to_numpy()[0]
        Ct = self._farm_results[FV.CT].to_numpy()[0]
        et = wd2uv(self._farm_results[FV.WD].to_numpy()[0], axis=-1)
        
        return St, Ct, et
        
    def wake_deficit(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Product over all wake functions (1-W_k), calculated on the given subgrid.

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
        """

        results = self._algo.calc_points(self._farm_results, subgrid.locations[None], outputs=[FV.AMB_WS, FV.WS])
        results = results[FV.WS].to_numpy()[0] / results[FV.AMB_WS].to_numpy()[0]
        return results.reshape(subgrid.Nx, subgrid.Ny, subgrid.Nz)
        
    def get_u_subgrid(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculate the wake model velocity field on the given subgrid.

        For uni-directional self-similar wake model, a base implementation is provided that uses the wake deficit shape
        functions.
        """
        point_results = self._algo.calc_points(self._farm_results, subgrid.locations[None], outputs=[FV.WS, FV.WD])
        uv = wd2uv(point_results[FV.WD].to_numpy()[0], point_results[FV.WS].to_numpy()[0])
        uv = uv.reshape(subgrid.Nx, subgrid.Ny, subgrid.Nz, 2)
        
        return uv[..., 0], uv[..., 1]
    
    def background_flow_direction(self, wind_farm, abl):
        """
        Return the direction of the flow according to the wake model, which is assumed to only depend on the unperturbed
        background flow defined in the given ABL object.
        """
        # Get average turbine
        turbines = wind_farm.turbines
        z_h = np.mean([turbine.zh for turbine in turbines])     # Turbine hub height

        # Wind speed at hub height
        u = abl.u(z_h)
        v = abl.v(z_h)

        # Get wind direction at hub height
        from wayve.forcing.wind_farms.wake_model_coupling.wake_models.wake_model_tools import e_spanwise
        from wayve.forcing.forcing_tools import e_streamwise
        e_str = e_streamwise(u, v)      # Unit vector along the wind direction
        e_span = e_spanwise(u, v)       # Unit vector in cross wind direction

        return e_str, e_span
        
    def xy_plane(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, xs, ys, z):
        """
        Compute an xy cross-section of the flow.
        """

        # Get grid information
        Nx = len(xs)
        Ny = len(ys)
        Nz = 1

        # Set up meshgrid
        zs = np.array([z])
        x_m, y_m, z_m = np.meshgrid(xs, ys, zs, indexing="ij")

        # Get background velocities #
        # SubGrid coordinates
        x_locs = np.ravel(x_m)
        y_locs = np.ravel(y_m)
        z_locs = np.ravel(z_m)
        locations = np.stack([x_locs, y_locs, z_locs], axis=1)
        del x_locs, y_locs, z_locs
        
        # run computations
        point_results = self._algo.calc_points(self._farm_results, locations[None], outputs=[FV.AMB_WS, FV.WS, FV.WD])
        amb_uv = wd2uv(point_results[FV.WD].to_numpy()[0], point_results[FV.AMB_WS].to_numpy()[0])
        uv = wd2uv(point_results[FV.WD].to_numpy()[0], point_results[FV.WS].to_numpy()[0])
        del point_results, locations
        
        # pick z = 0 for 2D slice
        amb_uv = amb_uv.reshape(Nx, Ny, Nz, 2)[:, :, 0]
        uv = uv.reshape(Nx, Ny, Nz, 2)[:, :, 0]
        return amb_uv[:, :, 0], amb_uv[:, :, 1], uv[:, :, 0], uv[:, :, 1]
    