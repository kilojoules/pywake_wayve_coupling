#!/usr/bin/env python

'''
Grid module
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "January 13, 2022"

import numpy as np

from wayve.grid import fft_routines


class Stat2Dgrid:
    '''
    Grid for two-dimensional, stationary simulation

    Data structure containing information about the numerical domain and the discretization
    '''

    def __init__(self, Lx, Nx, Ly, Ny):
        '''
        Parameters
        ----------
        Lx,Ly: float
            Length of the numerical domain (dimension 0 and 1)
        Nx,Ny: int
            Number of grid points (dimension 0 and 1)
        '''
        # Don't allow uneven grid sizes,
        # it is implicitly assumed that there are defunct modes
        assert np.mod(Nx, 2) == 0, 'Error, current implementation only allows even Nx'
        assert np.mod(Ny, 2) == 0, 'Error, current implementation only allows even Ny'
        # Set up grid parameters
        self.__Lx = Lx
        self.__Nx = Nx
        self.__Ly = Ly
        self.__Ny = Ny
        self.__Nx2 = int(Nx / 2) + 1  # zero freq, positive freq and defunct mode

    def deal_grid(self):
        '''
        Dealiasing grid

        Returns
        -------
        _: Stat2Dgrid
            Dealiasing grid with size = 3/2 original grid size
        '''
        assert np.mod(self.Nx, 4) == 0, 'Error, Nx is not a multiple of 4 so dealiasing grid is uneven'
        assert np.mod(self.Ny, 4) == 0, 'Error, Ny is not a multiple of 4 so dealiasing grid is uneven'
        # Make sure dealising grid has even number of grid point,
        return Stat2Dgrid(self.Lx, int(3 * self.Nx / 2), self.Ly, int(3 * self.Ny / 2))

    def c2r(self, cfield, returnErr=False):
        '''
        Perform an inverse Fourier transform.

        Parameters
        ----------
        cfield: 2d numpy array
            complex field, assuming
                - Hermitian symmetry
                - zero-wavenumber component at the center of the spectrum
        returnErr (optional): bool
            flag to return the maximum imaginary part of the real field (default: False)

        Returns
        -------
        rfield: 2d numpy array
            inverse Fourier transform of cfield (real part)
        err (optional): float
            maximum imaginary part of rfield (zero if input is Hermitian-symmetric)
        '''
        # cfield has dimension (Nx/2+1)*Ny. To get a real field with dimension Nx*Ny
        # I used irfft2
        rfield = fft_routines.c2r(cfield, self.N)

        if returnErr:
            err = np.amax(np.abs(np.imag(rfield)))
            out = (np.real(rfield), err)
        else:
            out = np.real(rfield)
        return out

    def r2c(self, rfield):
        '''
        Perform a Fourier transform

        Parameters
        ----------
        rfield: 2d numpy array
            real field

        Returns
        -------
        cfield: 2d numpy array
            Fourier transform of rfield
            (zero-wavenumber at the center of the spectrum)
        '''
        # return only zero mode (firt row), positive frequencies and defunt mode
        # which is located in the last row
        return fft_routines.r2c(rfield, self.N)

    def c2r_deal(self, cfield_shifted):
        '''
        Perform an inverse Fourier transform from complex to dealiasing space
        (complex field of size 3/2Nx*3/2Ny is obtained by zero-padding)

        Parameters
        ----------
        cfield_shifted: 2d numpy array
            complex field, assuming
                - Hermitian symmetry
                - zero-wavenumber component at the center of the spectrum

        Returns
        -------
        rfield32: 2d numpy array
            inverse Fourier transform of cfield32 in dealiasing space (real part)
        '''
        # Set up de-aiasing grid
        grid_32 = self.deal_grid()
        # Get rfield
        rfield2 = fft_routines.c2r_deal(cfield_shifted, self.Nx, self.Nx2, self.Ny,
                                        grid_32.Nx, grid_32.Ny, grid_32.N)
        return rfield2

    def r2c_deal(self, rfield32):
        '''
        Perform a Fourier transform from dealiasing to complex space
        (complex field of size Nx*Ny is obtained by disregarding high wavenumbers)

        Parameters
        ----------
        rfield32: 2d numpy array
            real field in dealiasing space

        Returns
        -------
        cfield: 2d numpy array
            Fourier transform of rfield32, disregarding high wavenumbers
            (zero-wavenumber component at the center of the spectrum)
        '''
        # Set up de-aiasing grid
        grid_32 = self.deal_grid()
        # Get cfield
        cfield_shifted = fft_routines.r2c_deal(rfield32, self.Nx2, self.Ny,
                                               grid_32.Nx, grid_32.Ny, grid_32.N)
        return cfield_shifted

    def defunct_modes(self, x):
        """
        Set the defunct modes of a given array (Nx2 by Ny) to 0.
        Parameters
        ----------
        x: numpy array
            Complex field (2D)
        """
        # Set defunct modes to zero
        defunct_k = [i + (self.Nx2 - 1) * self.Ny for i in range(self.Ny)]
        defunct_l = [i * self.Ny for i in range(0, self.Nx2)]
        defunctindices = np.array(defunct_l + defunct_k)
        x[defunctindices] = 0.
        return

    @property
    def N(self):
        '''Total number of grid points with N=Nx*Ny'''
        return self.Nx * self.Ny

    @property
    def N2(self):
        '''Total number of grid points with N=(Nx/2+1)*Ny'''
        return self.Nx2 * self.Ny

    @property
    def shape(self):
        '''Dimensions of the numerical domain given by Nx*Ny'''
        return self.Nx, self.Ny

    @property
    def shape2(self):
        '''Dimensions of the numerical domain given by Nx/2*Ny'''
        return self.Nx2, self.Ny

    @property
    def Lx(self):
        '''Length of the numerical domain in dimension 0'''
        return self.__Lx

    @property
    def Nx(self):
        '''Number of grid points in dimension 0'''
        return self.__Nx

    @property
    def Nx2(self):
        '''Half the number +1 of grid points in dimension 0'''
        return self.__Nx2

    @property
    def dx(self):
        '''Grid spacing in dimension 0'''
        return self.Lx / self.Nx

    @property
    def Ly(self):
        '''Length of the numerical domain in dimension 1'''
        return self.__Ly

    @property
    def Ny(self):
        '''Number of grid points in dimension 1'''
        return self.__Ny

    @property
    def dy(self):
        '''Grid spacing in dimension 1'''
        return self.Ly / self.Ny

    @property
    def xs(self):
        '''Array with grid points in dimension 0 (real space)'''
        return np.linspace(-self.Lx/2, self.Lx/2, self.Nx, endpoint=False)

    @property
    def ys(self):
        '''Array with grid points in dimension 1 (real space)'''
        return np.linspace(-self.Ly/2, self.Ly/2, self.Ny, endpoint=False)

    @property
    def ks(self):
        '''Array with wave numbers in dimension 0 (Fourier space)'''
        # Use built-in function fftfreq, which works for both even and uneven Nx
        # Also shift wavenumbers so that k=0 is at Nx/2
        return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.Nx, self.dx))

    @property
    def ks2(self):
        '''Array with wave numbers in dimension 0 (Fourier space)'''
        # zero mode, positive frequencies and defunct mode, which is the first
        # wavenumber in ks (ks[0])
        return np.concatenate([self.ks[(self.Nx2 - 1):], np.array([self.ks[0]])])

    @property
    def ls(self):
        '''Array with wave numbers in dimension 1 (Fourier space)'''
        # Use built-in function fftfreq, which works for both even and uneven Ny
        # Also shift wavenumbers so that l=0 is at Ny/2
        return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.Ny, self.dy))