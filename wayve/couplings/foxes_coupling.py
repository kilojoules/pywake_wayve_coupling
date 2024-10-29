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
        self._setup_foxes(wind_farm, u_bg_evaluator, abl)
        
        
        reset_engine()
        with Engine.new("numpy", chunk_size_states=1, chunk_size_points=100000, verbosity=1):
            self._farm_results = self._algo.calc_farm()
            point_results = self._algo.calc_points(self._farm_results, subgrid.locations[None])
        
        raise NotImplementedError(f"LOCATIONS {subgrid.locations.shape}")
        