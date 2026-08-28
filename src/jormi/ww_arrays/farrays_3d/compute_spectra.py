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
from jormi.ww_arrays.farrays_3d import farray_types

##
## === PUBLIC FUNCTIONS
##


def compute_power_spectrum_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> NDArray[Any]:
    """
    Compute the 3D power spectrum of a field array whose trailing 3 axes are the spatial
    grid (num_x0_cells, num_x1_cells, num_x2_cells), preceded by zero or more leading
    component axes, e.g. () for a scalar, (3,) for a vector, (3, 3) for a rank-2 tensor.
    """
    return _compute_spectra.compute_power_spectrum_farray(
        farray=farray_3d,
        resolution=resolution_3d,
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D field array of any rank."""
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray=farray_3d,
        resolution=resolution_3d,
    )


def compute_isotropic_power_spectrum_sarray(
    *,
    sarray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D scalar array."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_3d=sarray_3d,
        resolution_3d=resolution_3d,
    )


def compute_isotropic_power_spectrum_varray(
    *,
    varray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D vector array."""
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_3d=varray_3d,
        resolution_3d=resolution_3d,
    )


## } MODULE
