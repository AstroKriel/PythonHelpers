## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays import _compute_spectra
from jormi.ww_validation import validate_arrays, validate_types

##
## === PUBLIC FUNCTIONS
##


def compute_bandpass_filtered_farray(
    *,
    farray: NDArray[Any],
    resolution: tuple[int, ...],
    num_ranks: int,
    k_min: float,
    k_max: float,
) -> NDArray[Any]:
    """
    Band-pass filter a field array in Fourier space around [k_min, k_max]: a hard
    indicator mask over [k_min, k_max], an exact shell decomposition, but its real-space
    kernel is a long-range, oscillatory sinc, so the filtered field can ring past where
    the true field is zero.

    The trailing `len(resolution)` axes are the spatial grid, preceded by `num_ranks`
    leading component axes, e.g. 0 for a scalar, 1 for a vector, 2 for a rank-2 tensor.
    The same FFT kernel serves any rank and any number of spatial dimensions; a component
    axis is never masked separately from any other.
    """
    validate_arrays.ensure_array(
        array=farray,
        param_name="<farray>",
    )
    validate_types.ensure_tuple_of_ints(
        param=resolution,
        param_name="<resolution>",
        allow_none=False,
    )
    validate_types.ensure_finite_int(
        param=num_ranks,
        param_name="<num_ranks>",
        require_positive=True,
    )
    num_spatial_dims = len(resolution)
    if farray.ndim != num_ranks + num_spatial_dims:
        raise ValueError(
            "compute_bandpass_filtered_farray expects `farray.ndim == num_ranks + len(resolution)`:"
            f" got farray.ndim={farray.ndim}, num_ranks={num_ranks}, resolution={resolution}.",
        )
    if farray.shape[-num_spatial_dims:] != resolution:
        raise ValueError(
            "compute_bandpass_filtered_farray expects `farray.shape` to end with"
            f" `resolution`: got shape={farray.shape}, resolution={resolution}.",
        )
    _compute_spectra._ensure_isotropic_resolution(
        resolution,
        param_name="compute_bandpass_filtered_farray",
    )
    validate_types.ensure_ordered_pair(
        param=(k_min, k_max),
        param_name="<k_min, k_max>",
        allow_none=False,
    )
    if k_min < 0.0:
        raise ValueError(f"`k_min` must be non-negative, got {k_min}.")
    spatial_axes = tuple(range(-num_spatial_dims, 0))
    shifted_fft_farray = numpy.fft.fftshift(
        numpy.fft.fftn(
            farray,
            axes=spatial_axes,
            norm="forward",
        ),
        axes=spatial_axes,
    )
    k_magnitude = _compute_spectra._compute_radial_k_magnitude(
        num_cells_per_dim=resolution,
    )
    k_mask = (k_magnitude >= k_min) & (k_magnitude <= k_max)
    ## mask in place: no separate masked-copy kept alive alongside the unmasked FFT
    shifted_fft_farray *= k_mask
    filtered_fft_farray = numpy.fft.ifftshift(
        shifted_fft_farray,
        axes=spatial_axes,
    )
    del shifted_fft_farray
    filtered_farray = numpy.fft.ifftn(
        filtered_fft_farray,
        axes=spatial_axes,
        norm="forward",
    )
    del filtered_fft_farray
    ## `.real` on a complex array is a view, not a copy: left as-is, it would keep the
    ## full complex ifftn buffer (2x the memory) alive for as long as the caller holds it
    return numpy.ascontiguousarray(filtered_farray.real)


## } MODULE
