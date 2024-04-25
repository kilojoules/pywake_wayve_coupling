#!/usr/bin/env python

'''
Atmospheric Boundary Layer module

Module defining data structures for the atmospheric state
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "December 21, 2020"

import numpy as np
from numpy.lib.scimath import sqrt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

from wayve.abl.abl_tools import height_average


class ABL(object):
    '''
    Atmospheric state

    Data structure containing all information about the atmospheric state
    '''

    def __init__(self,
                 zs, us, vs, ths, tauxs, tauys,
                 H1, H2,
                 gprime, N, U3, V3,
                 fc,
                 nus=None,
                 rho=1.225, TI=0.04, z0=None, ust=None,
                 inv_bottom=None, inv_top=None,
                 h_strat=10.e3, Uinf=None, Vinf=None, Ninf=None):
        '''
        Initialise atmospheric state with the given information.

        An ABL object is initialised with the given data. The profiles (us, vs, ths, tauxs, tauys) are height-averaged
        over the given layer depths. Some optional parameters can be specified, but can also be calculated from the
        given profiles.

        The profiles are assumed to be ordered in increasing altitude (zs), and are all assumed to have the same length
        as zs.

        Parameters
        ----------
        zs: 1d numpy array
            Altitudes at which vertical profiles are defined (assumed sorted lowest to highest)
        us,vs: 1d numpy array
            Velocity profiles in x and y directions
        ths: 1d numpy array
            Potential temperature profile
        tauxs,tauys: 1d numpy array
            Momentum flux profiles in x and y direction
        H1,H2: float
            Thicknesses of the wind-farm and upper layers
        gprime: float
            Capping inversion reduced gravity
        N: float
            Brunt-Vaisala frequency
        U3,V3: float
            Velocities in the free atmosphere
        fc: float
            Coriolis parameter
        rho (optional): float
            Air density (default: 1.225kg/m3)
        TI (optional): float
            Turbulence intensity in the wind farm layer (default: 4%)
        z0 (optional): float
            Surface roughness (default: calculated from profiles)
        ust (optional): float
            Friction velocity (default: taken from the first elements of the momentum flux profiles)
        inv_bottom,inv_top (optional): float
            Bottom and top of the capping inversion (default: CI height)
        h_strat (optional): float
            Tropopause altitude (default: 10km)
        Uinf,Vinf,Ninf (optional): float
            Atmospheric state above the domain of interest (default: FA state U3,V3,N)
        '''
        # Universal constants #
        kappa = 0.41
        gravity = 9.80665
        # Scalar parameters #
        # Layer thickness
        self.__H1 = H1
        self.__H2 = H2
        H = H1 + H2
        # Velocity (x)
        self.__U1 = height_average(us, zs, 0., H1)
        self.__U2 = height_average(us, zs, H1, H)
        self.__U3 = U3
        # Velocity (y)
        self.__V1 = height_average(vs, zs, 0., H1)
        self.__V2 = height_average(vs, zs, H1, H)
        self.__V3 = V3
        # Stratification parameters
        self.__gprime = gprime
        self.__N = N
        # Coriolis parameter
        self.__fc = fc
        # Turbulence intensity
        self.__TI = TI
        # Profile arrays #
        # Altitude arrays
        self.__zs = zs
        self.__zst = zs
        # Velocity profiles
        self.__us = us
        self.__vs = vs
        # Potential temperature
        self.__ths = ths
        # Momentum fluxes
        self.__tauxs = tauxs
        self.__tauys = tauys
        # Post-processing and optional calculations #
        # Eddy viscosities
        if nus is None:     # Estimate eddy viscosity
            # Total velocity and momentum flux profiles
            Ms = np.sqrt(np.square(us) + np.square(vs))
            taus = np.sqrt(np.square(tauxs) + np.square(tauys))
            # Sheer profile
            ds = np.gradient(Ms, zs, edge_order=2)
            # Estimate the turbulent boundary layer height as the point where sheer becomes negative
            blh = np.min(zs[ds <= 0.])
            # Simple eddy viscosity within the boundary layer
            nus = np.divide(taus, ds, out=np.zeros_like(zs), where=(zs < blh))  # nu=0 for zs>=blh
        self.__nus = nus
        self.__nu1 = height_average(nus, zs, 0., H1)
        self.__nu2 = height_average(nus, zs, H1, H)
        # Brunt-Vaisala frequency
        theta_spline = UnivariateSpline(self.zs, self.ths, k=1, s=0, ext=0)     # Spline interpolation for derivatives
        dthetadz_spline = theta_spline.derivative()
        self.__Ns = np.real(sqrt(gravity*dthetadz_spline(self.zs)/theta_spline(self.H)))
        # Air density
        self.__rho = rho
        # Friction velocity
        if ust is not None:
            self.__utau = ust
        else:
            self.__utau = self.taust[0]
        # Surface roughness
        if z0 is not None:
            self.__z0 = z0
        else:
            self.__z0 = self.zs[0] / np.exp(kappa*self.Ms[0]/self.utau)
        # Capping inversion parameters
        if inv_bottom is not None:
            self.__inv_bottom = inv_bottom
        else:
            self.__inv_bottom = self.H
        if inv_top is not None:
            self.__inv_top = inv_top
        else:
            self.__inv_top = self.H
        # State at infinity
        self.__h_strat = h_strat
        if Uinf is not None:
            self.__Uinf = Uinf
        else:
            self.__Uinf = U3
        if Vinf is not None:
            self.__Vinf = Vinf
        else:
            self.__Vinf = V3
        if Ninf is not None:
            self.__Ninf = Ninf
        else:
            self.__Ninf = N

    @staticmethod
    def fromfile(filename):
        '''
        Load ABL object from file

        Parameters
        ----------
        filename: str
            File containing the ABL object
        '''
        # Read from file
        with open(filename, 'r') as file:

            # 7 header lines containing no information
            for _ in range(7):
                file.readline()

            # Scalar data
            H1         = float(file.readline().rstrip('\r\n').split('=')[1])
            U1         = float(file.readline().rstrip('\r\n').split('=')[1])
            V1         = float(file.readline().rstrip('\r\n').split('=')[1])
            nu1        = float(file.readline().rstrip('\r\n').split('=')[1])
            H2         = float(file.readline().rstrip('\r\n').split('=')[1])
            U2         = float(file.readline().rstrip('\r\n').split('=')[1])
            V2         = float(file.readline().rstrip('\r\n').split('=')[1])
            nu2        = float(file.readline().rstrip('\r\n').split('=')[1])
            U3         = float(file.readline().rstrip('\r\n').split('=')[1])
            V3         = float(file.readline().rstrip('\r\n').split('=')[1])
            gprime     = float(file.readline().rstrip('\r\n').split('=')[1])
            N          = float(file.readline().rstrip('\r\n').split('=')[1])
            fc         = float(file.readline().rstrip('\r\n').split('=')[1])
            TI         = float(file.readline().rstrip('\r\n').split('=')[1])
            z0         = float(file.readline().rstrip('\r\n').split('=')[1])
            utau       = float(file.readline().rstrip('\r\n').split('=')[1])
            rho        = float(file.readline().rstrip('\r\n').split('=')[1])
            inv_bottom = float(file.readline().rstrip('\r\n').split('=')[1])
            inv_top    = float(file.readline().rstrip('\r\n').split('=')[1])
            h_strat    = float(file.readline().rstrip('\r\n').split('=')[1])
            Uinf       = float(file.readline().rstrip('\r\n').split('=')[1])
            Vinf       = float(file.readline().rstrip('\r\n').split('=')[1])
            Ninf       = float(file.readline().rstrip('\r\n').split('=')[1])
            file.readline()

            # Cell-centered data
            for _ in range(2):
                file.readline()
            Nzs = int(file.readline().rstrip('\r\n').split('=')[1])
            file.readline()
            dummy = []
            for _ in range(Nzs):
                dummy.append([float(i) for i in file.readline().rstrip('\r\n').split(',')])
            data = np.array(dummy)
            zs = data[:, 0]
            us = data[:, 1]
            vs = data[:, 2]
            ths = data[:, 3]
            tauxs = data[:, 4]
            tauys = data[:, 5]
            nus = data[:, 6]
            file.readline()

        # Set up ABL object
        abl = ABL(zs, us, vs, ths, tauxs, tauys,
                  H1, H2,
                  gprime, N, U3, V3,
                  fc,
                  nus=nus, rho=rho, TI=TI, z0=z0, ust=utau,
                  inv_bottom=inv_bottom, inv_top=inv_top,
                  h_strat=h_strat, Uinf=Uinf, Vinf=Vinf, Ninf=Ninf)

        return abl

    def saveas(self, filename, info=''):
        '''
        Save ABL object to file

        Parameters
        ----------
        filename: str
            Name of destination file
        info (optional): str
            String with information about the ABL object (should not contain line breaks)
            (default: empty string)
        '''
        with open(filename, 'w') as file:
            # Write header
            file.write('%%%%%%%%%%%%%%\n')
            file.write('APM ABL object\n')
            file.write('%%%%%%%%%%%%%%\n')
            file.write(info + '\n')
            file.write('\n')
            # Scalar data
            file.write('1. Scalar data\n')
            file.write('--------------\n')
            file.write('H1         = ' + '{:17.10g}'.format(self.H1) + '\n')
            file.write('U1         = ' + '{:17.10g}'.format(self.U1) + '\n')
            file.write('V1         = ' + '{:17.10g}'.format(self.V1) + '\n')
            file.write('nu1        = ' + '{:17.10g}'.format(self.nu1) + '\n')
            file.write('H2         = ' + '{:17.10g}'.format(self.H2) + '\n')
            file.write('U2         = ' + '{:17.10g}'.format(self.U2) + '\n')
            file.write('V2         = ' + '{:17.10g}'.format(self.V2) + '\n')
            file.write('nu2        = ' + '{:17.10g}'.format(self.nu2) + '\n')
            file.write('U3         = ' + '{:17.10g}'.format(self.U3) + '\n')
            file.write('V3         = ' + '{:17.10g}'.format(self.V3) + '\n')
            file.write('gprime     = ' + '{:17.10g}'.format(self.gprime) + '\n')
            file.write('N          = ' + '{:17.10g}'.format(self.N) + '\n')
            file.write('fc         = ' + '{:17.10g}'.format(self.fc) + '\n')
            file.write('TI         = ' + '{:17.10g}'.format(self.TI) + '\n')
            file.write('z0         = ' + '{:17.10g}'.format(self.z0) + '\n')
            file.write('utau       = ' + '{:17.10g}'.format(self.utau) + '\n')
            file.write('rho        = ' + '{:17.10g}'.format(self.rho) + '\n')
            file.write('inv_bottom = ' + '{:17.10g}'.format(self.inv_bottom) + '\n')
            file.write('inv_top    = ' + '{:17.10g}'.format(self.inv_top) + '\n')
            file.write('h_strat    = ' + '{:17.10g}'.format(self.h_strat) + '\n')
            file.write('Uinf       = ' + '{:17.10g}'.format(self.Uinf) + '\n')
            file.write('Vinf       = ' + '{:17.10g}'.format(self.Vinf) + '\n')
            file.write('Ninf       = ' + '{:17.10g}'.format(self.Ninf) + '\n')
            file.write('\n')
            # Cell-centered data
            file.write('2. Profile data\n')
            file.write('---------------------\n')
            file.write('Nzs = ' + '{:3d}'.format(self.zs.size) + '\n')
            file.write('zs,us,vs,ths,tauxs,tauys,nus\n')
            for i in range(self.zs.size):
                file.write('{:17.10g}'.format(self.zs[i]) + ',' +
                           '{:17.10g}'.format(self.us[i]) + ',' +
                           '{:17.10g}'.format(self.vs[i]) + ',' +
                           '{:17.10g}'.format(self.ths[i]) + ',' +
                           '{:17.10g}'.format(self.tauxs[i]) + ',' +
                           '{:17.10g}'.format(self.tauys[i]) + ',' +
                           '{:17.10g}'.format(self.nus[i]) +
                           '\n')

    def rotate(self, alpha):
        '''
        Rotate coordinate axis over a certain angle

        Parameters
        ----------
        alpha: float
            Angle over which the coordinate axis is to be rotated (in radians)
        '''
        # Rotate height-averaged velocities
        U1n = self.U1 * np.cos(alpha) + self.V1 * np.sin(alpha)
        V1n = self.V1 * np.cos(alpha) - self.U1 * np.sin(alpha)
        self.__U1 = U1n
        self.__V1 = V1n
        U2n = self.U2 * np.cos(alpha) + self.V2 * np.sin(alpha)
        V2n = self.V2 * np.cos(alpha) - self.U2 * np.sin(alpha)
        self.__U2 = U2n
        self.__V2 = V2n
        U3n = self.U3 * np.cos(alpha) + self.V3 * np.sin(alpha)
        V3n = self.V3 * np.cos(alpha) - self.U3 * np.sin(alpha)
        self.__U3 = U3n
        self.__V3 = V3n
        # Rotate stratosphere
        Uinfn = self.Uinf * np.cos(alpha) + self.Vinf * np.sin(alpha)
        Vinfn = self.Vinf * np.cos(alpha) - self.Uinf * np.sin(alpha)
        self.__Uinf = Uinfn
        self.__Vinf = Vinfn
        # Rotate underlying data points
        un = self.us * np.cos(alpha) + self.vs * np.sin(alpha)
        vn = self.vs * np.cos(alpha) - self.us * np.sin(alpha)
        self.__us = un
        self.__vs = vn
        # Rotate momentum fluxes
        tauxn = self.tauxs * np.cos(alpha) + self.tauxs * np.sin(alpha)
        tauyn = self.tauys * np.cos(alpha) - self.tauys * np.sin(alpha)
        self.__tauxs = tauxn
        self.__tauys = tauyn

    def set_straight(self):
        '''
        Align the velocity profile with the x-axis
        '''
        # Align height-averaged velocities
        self.__U1 = self.S1
        self.__V1 = 0.
        self.__U2 = self.S2
        self.__V2 = 0.
        self.__U3 = self.S3
        self.__V3 = 0.
        # Align underlying data points
        sn = np.sqrt(np.power(self.us, 2) + np.power(self.vs, 2))
        self.__us = sn
        self.__vs = 0. * sn
        # Rotate momentum fluxes
        taun = np.sqrt(np.power(self.tauxs, 2) + np.power(self.tauys, 2))
        self.__tauxs = taun
        self.__tauys = 0. * taun
        return

    def add_FA(self, zs, us, vs, Ns, ths,
               U3, V3, N,
               h_strat, u_strat, v_strat, N_strat):
        """
        Add upper atmospheric profiles to this ABL.
        """
        # Get relevant profile range
        h_min = max(self.H, self.inv_top)
        self.h_strat = h_strat
        # Select part of new grid
        selection = np.logical_and(zs >= h_min, zs < self.h_strat)
        us = us[selection]
        vs = vs[selection]
        Ns = Ns[selection]
        ths = ths[selection]
        zs = zs[selection]
        # Remove FA from existing profiles
        selection_s = self.zs < h_min
        self.us = self.us[selection_s]
        self.vs = self.vs[selection_s]
        self.Ns = self.Ns[selection_s]
        self.ths = self.ths[selection_s]
        self.zs = self.zs[selection_s]
        self.tauxs = self.tauxs[selection_s]
        self.tauys = self.tauys[selection_s]
        # Add FA to ABL profiles
        self.us = np.append(self.us, us)
        self.vs = np.append(self.vs, vs)
        self.Ns = np.append(self.Ns, Ns)
        self.ths = np.append(self.ths, ths)
        self.zs = np.append(self.zs, zs)
        self.tauxs = np.append(self.tauxs, 0. * zs)
        self.tauys = np.append(self.tauys, 0. * zs)
        # Set up new uniform FA state
        self.U3 = U3
        self.V3 = V3
        self.N = N
        # Add stratosphere state
        self.Uinf = u_strat
        self.Vinf = v_strat
        self.Ninf = N_strat

    @property
    def us(self):
        '''Velocity profile in x-direction'''
        return self.__us

    @us.setter
    def us(self, value):
        self.__us = value

    def u(self, z):
        '''Function to evaluate u at desired heights, based on interpolation of us'''
        return np.interp(z, self.zs, self.us, left=0., right=self.Uinf)

    @property
    def vs(self):
        '''Velocity profile in y-direction'''
        return self.__vs

    @vs.setter
    def vs(self, value):
        self.__vs = value

    def v(self, z):
        '''Function to evaluate v at desired heights, based on interpolation of vs'''
        return np.interp(z, self.zs, self.vs, left=0., right=self.Vinf)

    @property
    def Ms(self):
        '''Velocity magnitude profile (for post-processing purposes)'''
        return np.sqrt(self.us ** 2 + self.vs ** 2)

    @property
    def ths(self):
        '''Potential temperature profile used to estimate
        temperature structure'''
        return self.__ths

    @ths.setter
    def ths(self, value):
        self.__ths = value

    @property
    def Ns(self):
        '''Brunt-Vaisala frequency profile'''
        return self.__Ns

    @Ns.setter
    def Ns(self, value):
        self.__Ns = value

    @property
    def taust(self):
        '''Shear stress profile used to estimate
        momentum transport coefficients'''
        return np.sqrt(np.power(self.tauxs, 2) + np.power(self.tauys, 2))

    @property
    def tauxs(self):
        '''Shear stress profile component in dimension 0 used to estimate
        momentum transport coefficients'''
        return self.__tauxs

    @tauxs.setter
    def tauxs(self, value):
        self.__tauxs = value

    @property
    def tauys(self):
        '''Shear stress profile component in dimension 1 used to estimate
        momentum transport coefficients'''
        return self.__tauys

    @tauys.setter
    def tauys(self, value):
        self.__tauys = value

    @property
    def utau(self):
        '''Friction velocity'''
        return self.__utau

    @property
    def zs(self):
        '''Height corresponding to the vertical profiles'''
        return self.__zs

    @zs.setter
    def zs(self, value):
        '''Height corresponding to the vertical profiles'''
        self.__zs = value

    @property
    def H1(self):
        '''Height of the wind-farm layer'''
        return self.__H1

    @property
    def U1(self):
        '''Height-averaged velocity in the wind-farm layer in dimension 0'''
        return self.__U1

    @U1.setter
    def U1(self, value):
        self.__U1 = value

    @property
    def V1(self):
        '''Height averaged velocity in the wind-farm layer in dimension 1'''
        return self.__V1

    @V1.setter
    def V1(self, value):
        self.__V1 = value

    @property
    def S1(self):
        '''Height-averaged velocity magnitude in the wind-farm layer'''
        return np.sqrt(self.U1 ** 2 + self.V1 ** 2)

    @property
    def WD1(self):
        '''
        Height-averaged wind direction in the wind-farm layer (degrees)

        Bug: np.arctan only recognises angles between -90 and +90
        Better would be to return np.arctan2(self.V1,self.U1)*180/np.pi
        Even better is to return the actual wind direction:
            return 180. + np.arctan2(self.U1,self.V1)*180/np.pi
        '''
        return np.arctan(self.V1 / self.U1) * 180 / np.pi

    @property
    def nus(self):
        '''Turbulent eddy viscosity profile'''
        return self.__nus

    @property
    def nu1(self):
        '''Height-averaged turbulent viscosity in the wind-farm layer'''
        return self.__nu1

    @nu1.setter
    def nu1(self, value):
        self.__nu1 = value

    @property
    def H2(self):
        '''Height of the upper layer'''
        return self.__H2

    @property
    def U2(self):
        '''Height-averaged velocity in the upper layer in dimension 0'''
        return self.__U2

    @U2.setter
    def U2(self, value):
        self.__U2 = value

    @property
    def V2(self):
        '''Height-averaged velocity in the upper layer in dimension 1'''
        return self.__V2

    @V2.setter
    def V2(self, value):
        self.__V2 = value

    @property
    def S2(self):
        '''Height-averaged velocity magnitude in the upper layer'''
        return np.sqrt(self.U2 ** 2 + self.V2 ** 2)

    @property
    def WD2(self):
        '''
        Height-averaged wind direction in the upper layer (degrees)

        Bug: see WD1
        '''
        return np.arctan(self.V2 / self.U2) * 180 / np.pi

    @property
    def nu2(self):
        '''Height-averaged turbulent viscosity in the upper layer'''
        return self.__nu2

    @nu2.setter
    def nu2(self, value):
        self.__nu2 = value

    @property
    def U3(self):
        '''Velocity in the free atmosphere in dimension 0'''
        return self.__U3

    @U3.setter
    def U3(self, value):
        self.__U3 = value

    @property
    def V3(self):
        '''Velocity in the free atmosphere in dimension 1'''
        return self.__V3

    @V3.setter
    def V3(self, value):
        self.__V3 = value

    @property
    def S3(self):
        '''Velocity magnitude in the free atmosphere'''
        return np.sqrt(self.U3 ** 2 + self.V3 ** 2)

    @property
    def WD3(self):
        '''
        Wind direction in the free atmosphere (degrees)

        Bug: see WD1
        '''
        return np.arctan(self.V3 / self.U3) * 180 / np.pi

    @property
    def dU12(self):
        '''
        Velocity difference between the wind-farm and upper layer in dimension 0
        '''
        return self.U1 - self.U2

    @property
    def dV12(self):
        '''
        Velocity difference between the wind-farm and upper layer in dimension 1
        '''
        return self.V1 - self.V2

    @property
    def dS12(self):
        '''
        Magnitude of velocity difference vector between the wind-farm and
        upper layer
        '''
        return np.sqrt(self.dU12 ** 2 + self.dV12 ** 2)

    @property
    def dS23(self):
        '''
        Magnitude of velocity difference vector between the upper layer and
        the free atmosphere
        '''
        return np.sqrt((self.U3 - self.U2) ** 2 + (self.V3 - self.V2) ** 2)

    @property
    def rho(self):
        '''Air density'''
        return self.__rho

    @property
    def z0(self):
        '''Surface roughness'''
        return self.__z0

    @property
    def gprime(self):
        '''Reduced gravity'''
        return self.__gprime

    @gprime.setter
    def gprime(self, value):
        self.__gprime = value

    @property
    def inv_bottom(self):
        '''Altitude of the bottom of the inversion layer'''
        return self.__inv_bottom

    @property
    def inv_top(self):
        '''Altitude of the top of the inversion layer'''
        return self.__inv_top

    @property
    def N(self):
        '''Brunt Vaisala frequency of the free atmosphere'''
        return self.__N

    @N.setter
    def N(self, value):
        self.__N = value

    @property
    def fc(self):
        '''Coriolis parameter'''
        return self.__fc

    @fc.setter
    def fc(self, value):
        self.__fc = value

    @property
    def TI(self):
        '''Streamwise turbulent intensity at hub height'''
        return self.__TI

    @TI.setter
    def TI(self, value):
        self.__TI = value

    @property
    def H(self):
        '''Boundary-layer height (= height of wind-farm plus upper layer)'''
        return self.H1 + self.H2

    @property
    def Ub(self):
        '''Boundary-layer velocity scale'''
        # Projection of (U2,V2) onto (U1,V1)
        U2p = (self.U1 * self.U2 + self.V1 * self.V2) / self.S1
        return (self.H1 / self.H * self.S1 ** (-2) + self.H2 / self.H * U2p ** (-2)) ** (-1 / 2)

    @property
    def PN(self):
        '''
        Non-dimensional number characterising internal gravity wave amplitude
        '''
        return self.Ub ** 2 / (self.N * self.S3 * self.H)

    @property
    def Fr(self):
        '''Froude number'''
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.Ub / np.sqrt(self.gprime * self.H)

    @property
    def Fr1(self):
        '''Partial Froude number of the wind-farm layer'''
        return self.S1 / np.sqrt(self.gprime * self.H1)

    @property
    def Fr2(self):
        '''Partial Froude number of the upper layer'''
        return self.S2 / np.sqrt(self.gprime * self.H2)

    @property
    def h_strat(self):
        '''Tropopause altitude, above which the upper atmosphere is considered uniform'''
        return self.__h_strat

    @h_strat.setter
    def h_strat(self, value):
        self.__h_strat = value

    @property
    def Ninf(self):
        '''Stratification in the stratosphere'''
        return self.__Ninf

    @Ninf.setter
    def Ninf(self, value):
        self.__Ninf = value

    @property
    def Uinf(self):
        '''Velocity in the stratosphere in dimension 0'''
        return self.__Uinf

    @Uinf.setter
    def Uinf(self, value):
        self.__Uinf = value

    @property
    def Vinf(self):
        '''Velocity in the stratosphere in dimension 1'''
        return self.__Vinf

    @Vinf.setter
    def Vinf(self, value):
        self.__Vinf = value

    @property
    def S_strat(self):
        '''Wind speed in the stratosphere'''
        return np.sqrt(self.Uinf ** 2 + self.Vinf ** 2)


