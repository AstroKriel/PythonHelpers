## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_arrays import _filter_spectra
from jormi.ww_arrays.farrays_3d import farray_types

##
## === PUBLIC FUNCTIONS
##


def compute_bandpass_filtered_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    num_ranks: int,
    k_min: float,
    k_max: float,
) -> NDArray[Any]:
    """Band-pass filter a 3D field array of any rank in Fourier space around [k_min, k_max]."""
    return _filter_spectra.compute_bandpass_filtered_farray(
        farray=farray_3d,
        resolution=resolution_3d,
        num_ranks=num_ranks,
        k_min=k_min,
        k_max=k_max,
    )


def compute_bandpass_filtered_sarray(
    *,
    sarray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    k_min: float,
    k_max: float,
) -> NDArray[Any]:
    """Band-pass filter a 3D scalar array in Fourier space."""
    farray_types.ensure_3d_sarray(
        sarray_3d=sarray_3d,
        param_name="<sarray_3d>",
    )
    return compute_bandpass_filtered_farray(
        farray_3d=sarray_3d,
        resolution_3d=resolution_3d,
        num_ranks=0,
        k_min=k_min,
        k_max=k_max,
    )


def compute_bandpass_filtered_varray(
    *,
    varray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    k_min: float,
    k_max: float,
) -> NDArray[Any]:
    """Band-pass filter a 3D vector array in Fourier space, one component at a time."""
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    return compute_bandpass_filtered_farray(
        farray_3d=varray_3d,
        resolution_3d=resolution_3d,
        num_ranks=1,
        k_min=k_min,
        k_max=k_max,
    )


def compute_bandpass_filtered_r2tarray(
    *,
    r2tarray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    k_min: float,
    k_max: float,
) -> NDArray[Any]:
    """Band-pass filter a 3D rank-2 tensor array in Fourier space, one component at a time."""
    farray_types.ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name="<r2tarray_3d>",
    )
    return compute_bandpass_filtered_farray(
        farray_3d=r2tarray_3d,
        resolution_3d=resolution_3d,
        num_ranks=2,
        k_min=k_min,
        k_max=k_max,
    )


## } MODULE
