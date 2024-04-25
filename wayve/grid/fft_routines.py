#!/usr/bin/env python

'''
File containing functions to execute the FFTs.
If numba ever supports the np.fft module, these should all be pre-compiled.
'''

__author__ = "Dries Allaerts, Luca Lanzilao, Koen Devesse"
__date__ = "February 15, 2022"

import numpy as np


def c2r(cfield, N):
    '''
    Perform an inverse Fourier transform.

    Parameters
    ----------
    cfield: 2d numpy array
        complex field, assuming
            - Hermitian symmetry
            - zero-wavenumber component at the center of the spectrum
    N: int
        grid points

    Returns
    -------
    rfield: 2d numpy array
        inverse Fourier transform of cfield (real part)
    '''
    # cfield has dimension (Nx/2+1)*Ny. To get a real field with dimension Nx*Ny
    # I used irfft2
    return np.fft.irfft2(np.fft.fftshift(cfield * N, axes=1), axes=(1, 0))


def r2c(rfield, N):
    '''
    Perform a Fourier transform

    Parameters
    ----------
    rfield: 2d numpy array
        real field
    N: int
        grid points

    Returns
    -------
    cfield: 2d numpy array
        Fourier transform of rfield
        (zero-wavenumber at the center of the spectrum)
    '''
    # return only zero mode (firt row), positive frequencies and defunt mode
    # which is located in the last row
    return np.fft.fftshift(np.fft.rfft2(rfield, axes=(1, 0)), axes=1) / N


def c2r_deal(cfield_shifted, Nx, Nx2, Ny, N32x, N32y, N32):
    '''
    Perform an inverse Fourier transform from complex to dealiasing space
    (complex field of size 3/2Nx*3/2Ny is obtained by zero-padding)

    Parameters
    ----------
    cfield_shifted: 2d numpy array
        complex field, assuming
            - Hermitian symmetry
            - zero-wavenumber component at the center of the spectrum
    Nx: int
        grid points in x-dimension
    Nx2: int
        half the number +1 of grid points in x-dimension
    Ny: int
        grid points in y-dimension
    N32x: int
        grid points of the de-aliased grid in x-dimension
    N32y: int
        grid points of the de-aliased grid in y-dimension
    N32: int
        grid points of the de-aliased grid

    Returns
    -------
    rfield32: 2d numpy array
        inverse Fourier transform of cfield32 in dealiasing space (real part)
    '''
    # First padd in x-direction
    # I padd only after last row, since at row zero I have the zero mode and
    # it doesnt make sense to pad before it
    cfield_shifted = np.concatenate((
        cfield_shifted,
        np.zeros((int(Nx / 4), Ny), dtype=np.complex128)),
        axis=0)
    # Then padd in y-direction
    # the matrix has Nx/2+1+Nx/4 rows at this point
    cfield_shifted = np.concatenate((
        np.zeros((int(Nx2 + Nx / 4), int(Ny / 4)), dtype=np.complex128),
        cfield_shifted,
        np.zeros((int(Nx2 + Nx / 4), int(Ny / 4)), dtype=np.complex128)),
        axis=1)
    cfield = np.fft.ifftshift(cfield_shifted, axes=1)
    rfield2 = np.fft.irfft2(cfield * N32, s=[N32y, N32x], axes=(1, 0))
    return rfield2


def r2c_deal(rfield32, Nx2, Ny, N32x, N32y, N32):
    '''
    Perform a Fourier transform from dealiasing to complex space
    (complex field of size Nx*Ny is obtained by disregarding high wavenumbers)

    Parameters
    ----------
    rfield32: 2d numpy array
        real field in dealiasing space
    Nx2: int
        half the number +1 of grid points in x-dimension
    Ny: int
        grid points in y-dimension
    N32x: int
        grid points of the de-aliased grid in x-dimension
    N32y: int
        grid points of the de-aliased grid in y-dimension
    N32: int
        grid points of the de-aliased grid

    Returns
    -------
    cfield: 2d numpy array
        Fourier transform of rfield32, disregarding high wavenumbers
        (zero-wavenumber component at the center of the spectrum)
    '''
    # same as r2c. I then retain only a matrix of dimension (Nx/2+1)*Ny
    cfield322 = np.fft.rfft2(rfield32, s=[N32y, N32x], axes=(1, 0)) / N32
    cfield_shifted = np.fft.fftshift(cfield322, axes=1)
    cfield_shifted = cfield_shifted[0:Nx2, int(Ny / 4):int(Ny * 5 / 4)]
    return cfield_shifted
