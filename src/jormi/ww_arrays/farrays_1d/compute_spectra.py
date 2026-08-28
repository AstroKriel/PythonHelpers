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
from jormi.ww_arrays.farrays_1d import farray_types

##
## === PUBLIC FUNCTIONS
##


def compute_power_spectrum_farray(
    *,
    farray_1d: NDArray[Any],
    resolution_1d: tuple[int],
    num_ranks: int,
) -> NDArray[Any]:
    """
    Compute the 1D power spectrum of a field array whose trailing axis is the spatial grid
    (num_x0_cells,), preceded by `num_ranks` leading component axes, e.g. 0 for a scalar,
    1 for a vector.
    """
    return _compute_spectra.compute_power_spectrum_farray(
        farray=farray_1d,
        resolution=resolution_1d,
        num_ranks=num_ranks,
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray_1d: NDArray[Any],
    resolution_1d: tuple[int],
    num_ranks: int,
) -> IsotropicPowerSpectrum:
    """
    Compute the 1D power spectrum of a field array of any rank, folded onto |k|.

    For 1D data there is no shell to integrate over (a "shell" at fixed |k| is just the pair
    of modes +k and -k), so this simply folds the negative-frequency half onto the
    positive-frequency half, same as `compute_isotropic_power_spectrum_farray` does in 2D/3D
    for a circular/spherical shell.
    """
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray=farray_1d,
        resolution=resolution_1d,
        num_ranks=num_ranks,
    )


def compute_isotropic_power_spectrum_sarray(
    *,
    sarray_1d: NDArray[Any],
    resolution_1d: tuple[int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D power spectrum of a 1D scalar array, e.g. a profile extracted from a 3D snapshot."""
    farray_types.ensure_1d_sarray(
        sarray_1d=sarray_1d,
        param_name="<sarray_1d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_1d=sarray_1d,
        resolution_1d=resolution_1d,
        num_ranks=0,
    )


def compute_isotropic_power_spectrum_varray(
    *,
    varray_1d: NDArray[Any],
    resolution_1d: tuple[int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D power spectrum of a 1D vector array."""
    farray_types.ensure_1d_varray(
        varray_1d=varray_1d,
        param_name="<varray_1d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_1d=varray_1d,
        resolution_1d=resolution_1d,
        num_ranks=1,
    )


## } MODULE
