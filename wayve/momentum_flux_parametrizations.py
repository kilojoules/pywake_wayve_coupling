'''
A file containing momentum flux parametrizations for the APM.
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "November 15, 2022"

import numpy as np


class MFP:
    """Generic abstract class to define a momentum flux parametrization interface"""

    def mf_term_Jac(self, abl):
        """The Jacobians of the vertical momentum flux terms in the APM momentum equations w.r.t. the APM variables.

        The Jacobian is returned as a 4x6 numpy array, where the columns indicate the relevant equation (x1, y1, x2,
        y2), and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2)."""
        # Initialize matrix
        jac = np.zeros((4, 6))
        # ABL variables
        H1 = abl.H1
        H2 = abl.H2
        # Get Jacobians of tau_0, tau_1
        tau_0_Jac = self.tau_0_Jac(abl)
        tau_1_Jac = self.tau_1_Jac(abl)
        # Add flux Jacobians to matrix
        jac[:2, :] += -tau_0_Jac / H1
        jac[:2, :] += tau_1_Jac / H1
        jac[2:, :] += -tau_1_Jac / H2
        # Add derivatives of total term w.r.t. h1 and h2, with tau constant #
        # Set up derivatives matrix
        D = np.zeros((2, 2))
        # Background momentum fluxes variables
        T = tau_interfaces(abl)
        # Calculate derivatives
        D[0, :] = - (T[1, :] - T[0, :]) / H1 ** 2
        D[1, :] = T[1, :] / H2 ** 2
        # Add to matrix
        jac[0, 4] += D[0, 0]
        jac[1, 4] += D[0, 1]
        jac[2, 5] += D[1, 0]
        jac[3, 5] += D[1, 1]
        # Return Jacobian
        return jac

    def tau_0_Jac(self, abl):
        """The Jacobian of the momentum exchange with the ground with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        raise Exception("Jacobian of tau_0 not implemented yet!")

    def tau_1_Jac(self, abl):
        """The Jacobian of the momentum exchange between the layers with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        raise Exception("Jacobian of tau_1 not implemented yet!")


class PotentialFlow(MFP):
    """A class for the momentum flux parametrization of potential flows, i.e. no momentum flux."""

    def mf_term_Jac(self, abl):
        """
        Zero matrices representing the Jacobians of the vertical momentum flux terms in the APM momentum equations.
        """
        # Initialize matrix
        jac = np.zeros((4, 6))
        return jac

    def tau_0_Jac(self, abl):
        """A zero matrix representing the Jacobian of the momentum exchange with the ground.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        All the elements are zero, since this represents potential flow.
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        return jac

    def tau_1_Jac(self, abl):
        """A zero matrix representing the Jacobian of the momentum exchange between the layers.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        All the elements are zero, since this represents potential flow.
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        return jac


class FrictionCoefficients(MFP):
    """A class for the basic momentum flux parametrization devised by Allaerts and Meyers (2019)."""

    def tau_0_Jac(self, abl):
        """The Jacobian of the momentum exchange with the ground with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        # Read out ABL state
        tau_vec = tau_interfaces(abl)[0, :]
        tau = np.sqrt(tau_vec[0]**2 + tau_vec[1]**2)
        S1 = abl.S1
        U1 = abl.U1
        V1 = abl.V1
        # Calculate coefficient
        C0 = tau / S1 ** 2
        # Calculate derivatives
        jac[0, 0] = C0*(S1**2+U1**2)/S1
        jac[0, 1] = C0*U1*V1/S1
        jac[1, 0] = C0*U1*V1/S1
        jac[1, 1] = C0*(S1**2+V1**2)/S1
        return jac

    def tau_1_Jac(self, abl):
        """The Jacobian of the momentum exchange between the layers with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        # Read out ABL state
        tau_vec = tau_interfaces(abl)[1, :]
        tau = np.sqrt(tau_vec[0]**2 + tau_vec[1]**2)
        dS12 = abl.dS12
        dU12 = abl.dU12
        dV12 = abl.dV12
        # Calculate coefficient
        C1 = tau / dS12 ** 2
        # Calculate Jacobian w.r.t. velocity differences #
        jac_du = np.zeros((2, 2))
        jac_du[0, 0] = C1*(dS12**2+dU12**2)/dS12
        jac_du[0, 1] = C1*dU12*dV12/dS12
        jac_du[1, 0] = C1*dU12*dV12/dS12
        jac_du[1, 1] = C1*(dS12**2+dV12**2)/dS12
        # Set up Jacobian w.r.t. layer velocities #
        # x-component
        jac[0, 0] = -jac_du[0, 0]
        jac[0, 1] = -jac_du[0, 1]
        jac[0, 2] = jac_du[0, 0]
        jac[0, 3] = jac_du[0, 1]
        # y-component
        jac[1, 0] = -jac_du[1, 0]
        jac[1, 1] = -jac_du[1, 1]
        jac[1, 2] = jac_du[1, 0]
        jac[1, 3] = jac_du[1, 1]
        return jac


class EddyViscosity(MFP):
    """
    A class for a basic momentum flux parametrization, where the momentum flux is assumed to scale directly with the
    velocity difference.
    """

    def tau_0_Jac(self, abl):
        """The Jacobian of the momentum exchange with the ground with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        # Read out ABL state
        tau_vec = tau_interfaces(abl)[0, :]
        tau = np.sqrt(tau_vec[0]**2 + tau_vec[1]**2)
        dS = abl.S1
        # Calculate coefficient
        C = tau / dS
        # Calculate derivatives
        jac[0, 0] = C
        jac[1, 1] = C
        return jac

    def tau_1_Jac(self, abl):
        """The Jacobian of the momentum exchange between the layers with respect to the APM variables.

        The Jacobian is returned as a 2x6 numpy array, where the columns indicate the momentum flux component (x and y),
        and the rows indicate the relevant APM variable (ordered as u1, v1, u2, v2, eta1, eta2).
        """
        # Initialize matrix
        jac = np.zeros((2, 6))
        # Read out ABL state
        tau_vec = tau_interfaces(abl)[1, :]
        tau = np.sqrt(tau_vec[0]**2 + tau_vec[1]**2)
        dS = abl.dS12
        # Calculate coefficient
        C = tau / dS
        # Calculate derivatives
        jac[0, 0] = -C
        jac[1, 1] = -C
        jac[0, 2] = C
        jac[1, 3] = C
        return jac


def tau_interfaces(abl):
    '''
    Evaluate the vertical turbulent momentum fluxes at the layer interfaces for the given ABL.

    Structured as a 4x4 numpy array::
        T[0,0] = T_xz at the ground
        T[0,1] = T_yz at the ground
        T[1,0] = T_xz at the interface between the lower and upper layer
        T[1,1] = T_yz at the interface between the lower and upper layer
    '''
    # Friction velocity
    ust = abl.utau
    # Velocities at ground level
    u0 = abl.us[0]
    v0 = abl.vs[0]
    m0 = np.sqrt(u0**2 + v0**2)
    # Output array
    T = np.zeros((2, 2))
    # Ground interface
    T[0, 0] = ust**2 * u0 / m0
    T[0, 1] = ust**2 * v0 / m0
    # Layer interface
    T[1, 0] = np.interp(abl.H1, abl.zs, abl.tauxs)
    T[1, 1] = np.interp(abl.H1, abl.zs, abl.tauys)
    return T
