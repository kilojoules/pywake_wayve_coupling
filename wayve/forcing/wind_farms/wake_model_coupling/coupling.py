"""
File containing the basic abstract interface for wake model coupling to the APM.
"""

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "February 4, 2022"


class Coupling:
    """
    An abstract class to set up an interface between the APM, and specific wake model coupling implementations. This
    base class only defines the minimum interface expected by the APM, and leaves any code interfacing to the wake model
    up to the subclasses.

    The coupling calculations are done using the methods "preprocess" and "reprocess". When these are called, the
    coupling object should run any calculations necessary to obtain the turbine inflow velocities (St), the thrust
    coefficients (Ct), and the turbine directions (et). It should store these, so that the APM can use them later. This
    avoids repeated calculations.

    It's worth noting that there can be multiple ways of coupling a wake model to the APM. In that case, simply
    subclassing is sufficient for now. See, for example, VaryingBackground and its subclasses, that can be used to
    couple wake models with heterogeneous background velocities.

    Finally, while some parametrizations, such as DispersiveStresses, require evaluating the wake model velocity on a
    dense grid, this basic interface does not provide that. This is because this is an expensive operation which should
    be avoided. Specific subclasses, such as VaryingBackground, can provide an implementation for this, but this is not
    expected of new coupling methods.
    """

    def __init__(self):
        """
        Set up Coupling object relating the APM structure to a wake model.
        """
        self.__St = None
        self.__et = None
        self.__Ct = None

    def preprocess(self, model, wind_farm):
        """
        Initial calculation of the wake model, assuming unperturbed background state.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        wind_farm   WindFarm object
                        Wind farm for which the calculation is performed
        """
        raise Exception("Preprocessing not implemented yet!")

    def reprocess(self, model, wind_farm, result):
        """
        Update the wake model, given the new APM state.

        Parameters
        ----------
        model   Model object
                    APM object, providing access to ABL, grid, and parametrization information
        wind_farm   WindFarm object
                        Wind farm for which the calculation is performed
        result      dict
                        State of the APM variables
        """
        raise Exception("Not implemented yet!")

    @property
    def St(self):
        """
        The inflow velocities of the turbines.
        """
        return self.__St

    @St.setter
    def St(self, value):
        self.__St = value

    @property
    def et(self):
        """
        The flow directions at the turbines.
        """
        return self.__et

    @et.setter
    def et(self, value):
        self.__et = value

    @property
    def Ct(self):
        """
        The thrust coefficients of the turbines.
        """
        return self.__Ct

    @Ct.setter
    def Ct(self, value):
        self.__Ct = value
