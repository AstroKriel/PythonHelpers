## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_arrays import _compute_spectra
from jormi.ww_arrays._compute_spectra import IsotropicPowerSpectrum as IsotropicPowerSpectrum
from jormi.ww_arrays.farrays_2d import farray_types

##
## === PUBLIC FUNCTIONS
##


def compute_power_spectrum_farray(
    *,
    farray_2d: NDArray[Any],
    resolution_2d: tuple[int, int],
) -> NDArray[Any]:
    """
    Compute the 2D power spectrum of a field array whose trailing 2 axes are the spatial
    grid (num_x0_cells, num_x1_cells), preceded by zero or more leading component axes,
    e.g. () for a scalar, (2,) for a vector.
    """
    return _compute_spectra.compute_power_spectrum_farray(
        farray=farray_2d,
        resolution=resolution_2d,
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray_2d: NDArray[Any],
    resolution_2d: tuple[int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 2D field array of any rank."""
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray=farray_2d,
        resolution=resolution_2d,
    )


def compute_isotropic_power_spectrum_sarray(
    *,
    sarray_2d: NDArray[Any],
    resolution_2d: tuple[int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 2D scalar array."""
    farray_types.ensure_2d_sarray(
        sarray_2d=sarray_2d,
        param_name="<sarray_2d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_2d=sarray_2d,
        resolution_2d=resolution_2d,
    )


def compute_isotropic_power_spectrum_varray(
    *,
    varray_2d: NDArray[Any],
    resolution_2d: tuple[int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 2D vector array."""
    farray_types.ensure_2d_varray(
        varray_2d=varray_2d,
        param_name="<varray_2d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_2d=varray_2d,
        resolution_2d=resolution_2d,
    )


## } MODULE
