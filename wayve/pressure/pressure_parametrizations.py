'''
A file containing pressure closure equations for the APM.
'''

__author__ = "Koen Devesse"
__date__ = "May 31, 2023"

import numpy as np


class PressureParametrization:
    """Generic abstract class to define a pressure parametrization interface"""

    def __init__(self):
        """
        Initialize the pressure parametrization.
        """
        # Computation results
        self.__Phi = None

    def preprocess(self, abl, grid):
        """
        Preprocess for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        # Free atmosphere
        self.Phi = self.get_Phi(abl, grid)

    def get_Phi(self, abl, grid):
        """
        Compute the stratification coefficients for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        raise Exception("Phi computation not implemented yet for this type of pressure parametrization!")

    @property
    def Phi(self):
        return self.__Phi

    @Phi.setter
    def Phi(self, value):
        self.__Phi = value


class NoPressureFeedback(PressureParametrization):
    """A class for the free atmosphere pressure parametrization where there is no pressure feedback to the APM state."""

    def get_Phi(self, abl, grid):
        """
        Compute the stratification coefficients for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        return np.zeros(grid.shape2, dtype=np.complex128)


class RigidLid(PressureParametrization):
    """A class for the free atmosphere pressure parametrization where the capping inversion is a rigid lid."""

    def get_Phi(self, abl, grid):
        """
        Compute the stratification coefficients for the given ABL and grid.

        Parameters
        ----------
        abl     ABL object
                    ABL of the APM
        grid    Grid object
                    Grid of the APM
        """
        # Take extremely high CI strength
        # Rigid lid corresponds to g_prime=inf, but this causes numerical issues with matrix inversions.
        # Therefore, we simply take a very high value.
        ci_strength = 1.e10
        return ci_strength * np.ones(grid.shape2, dtype=np.complex128)
