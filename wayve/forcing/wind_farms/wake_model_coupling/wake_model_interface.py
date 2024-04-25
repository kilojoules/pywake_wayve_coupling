"""
File containing an interface for wake models to implement when using the VaryingBackground coupling methods.
"""

__author__ = "Koen Devesse"
__date__ = "February 12, 2024"

import numpy as np


class WakeModelInterface:
    """
    A common interface for wake models, used by the VaryingBackground coupling methods.

    When coupling a wake model to the APM, these are the classes that should be implemented.
    """

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
        raise NotImplementedError("Wake model calculation not implemented yet!")

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
        """
        raise NotImplementedError("Velocity calculation not implemented yet!")


class UniDirectionalSelfSimilar(WakeModelInterface):
    """
    An interface for uni-directional self-similar wake models, used by the VelocityMatching coupling method.

    The velocity matching coupling method in the formulation of Devesse et al. (2023) requires the calculation of the
    wake deficit shape functions. Since this is not needed for other coupling methods, it uses this separate wake model
    interface.

    Additionally, this class can be used in combination with a SelfSimilarWMVH object. This specialized class of
    WakeModelVelocityHandler objects, defined in varying_background.py, stores the wake deficit shape functions, and
    only updates them when needed. This can drastically reduce the computational cost of parametrizations that require
    the subgrid velocities, such as DispersiveStresses (defined in dispersive_stresses.py).
    """

    def background_flow_direction(self, wind_farm, abl):
        """
        Return the direction of the flow according to the wake model, which is assumed to only depend on the unperturbed
        background flow defined in the given ABL object.
        """
        raise NotImplementedError("Flow direction calculation not implemented yet!")

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
        raise NotImplementedError("Wake deficit shape multiplier not implemented yet!")

    def get_u_subgrid(self, wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid):
        """
        Calculate the wake model velocity field on the given subgrid.

        For uni-directional self-similar wake model, a base implementation is provided that uses the wake deficit shape
        functions.
        """
        # Get wake multiplier
        w_multiplier = self.wake_deficit(wind_farm, abl, u_bg_evaluator, apm_evaluator, subgrid)
        # Apply wake multiplier to background velocity
        u_wm, v_wm = self.apply_wake_multiplier(wind_farm, abl, u_bg_evaluator, subgrid, w_multiplier)
        return u_wm, v_wm

    def apply_wake_multiplier(self, wind_farm, abl, u_bg_evaluator, subgrid, w_multiplier):
        """
        Applies the wake deficit shape functions to the background velocities to obtain the wake model velocities.
        """
        # Get background velocities #
        # SubGrid coordinates
        locations = subgrid.locations
        # Evaluate background velocities
        vel_bg = u_bg_evaluator(locations)
        # Reshape into grid
        Nx, Ny, Nz = subgrid.shape
        u_bg = np.reshape(vel_bg[:, 0], (Nx, Ny, Nz))
        v_bg = np.reshape(vel_bg[:, 1], (Nx, Ny, Nz))
        # Get streamwise direction
        e_str, e_span = self.background_flow_direction(wind_farm, abl)
        # Split up into streamwise and spanwise components
        str_bg = u_bg * e_str[0] + v_bg * e_str[1]
        span_bg = u_bg * e_span[0] + v_bg * e_span[1]
        # Add wakes
        str_wm = np.multiply(str_bg, w_multiplier)
        # Convert to u and v components
        u_wm = str_wm * e_str[0] + span_bg * e_span[0]
        v_wm = str_wm * e_str[1] + span_bg * e_span[1]
        return u_wm, v_wm
