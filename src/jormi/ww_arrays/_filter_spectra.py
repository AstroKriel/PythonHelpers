## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from typing import Any, Literal

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays import _compute_spectra
from jormi.ww_validation import validate_arrays, validate_types

##
## === DATA STRUCTURES
##

FilterWindow = Literal["tophat", "gaussian", "tukey"]


@dataclass(frozen=True)
class BandpassFilteredFFT:
    """
    The full Fourier-space state of a band-pass filter, alongside its real-space result.

    `shifted_fft_farray` and `shifted_fft_farray_masked` are fftshifted (zero-frequency
    centered at `resolution[i] // 2` along each spatial axis), matching the convention
    `_compute_spectra._compute_radial_k_magnitude` uses.
    """

    filtered_farray: NDArray[Any]
    shifted_fft_farray: NDArray[Any]
    shifted_fft_farray_masked: NDArray[Any]


##
## === INTERNAL HELPERS
##


def _compute_k_mask(
    *,
    k_magnitude: NDArray[Any],
    k_min: float,
    k_max: float,
    window: FilterWindow,
    taper_width: float | None = None,
) -> NDArray[Any]:
    """
    Build the k-space mask a band-pass filter multiplies onto the (shifted) FFT.

    `"tophat"` is a hard indicator over [k_min, k_max]: exact shell decomposition, but its
    real-space kernel is a long-range, oscillatory sinc, so the filtered field can ring and
    is not bounded by the original field's range. `"gaussian"` is a smooth bump centered on
    the same nominal band (`k0 = (k_min + k_max) / 2`, `sigma = (k_max - k_min) / 2`), with
    no hard edge anywhere, so it stays local and bounded, but it also has no hard *stop*:
    it always leaks some weight from outside [k_min, k_max]. `"tukey"` (needs `taper_width`)
    is flat at 1 across the interior of [k_min, k_max], raised-cosine tapered down to exactly
    0 over the outer `taper_width` of each edge, and exactly 0 outside the band: no leakage,
    and (unlike the tophat) both the mask's value and slope are continuous at the band edges,
    so it rings far less.
    """
    match window:
        case "tophat":
            return (k_magnitude >= k_min) & (k_magnitude <= k_max)
        case "gaussian":
            k_center = 0.5 * (k_min + k_max)
            k_width = 0.5 * (k_max - k_min)
            if not (k_width > 0.0):
                raise ValueError(
                    "a gaussian window needs `k_min < k_max` (strictly), so its width is"
                    f" positive; got k_min={k_min}, k_max={k_max}.",
                )
            return numpy.exp(-0.5 * numpy.square((k_magnitude - k_center) / k_width))
        case "tukey":
            if taper_width is None:
                raise ValueError(
                    "a tukey window needs `taper_width` (in the same k units as k_min/k_max).",
                )
            if not (taper_width > 0.0):
                raise ValueError(f"`taper_width` must be positive, got {taper_width}.")
            if not (2.0 * taper_width <= (k_max - k_min)):
                raise ValueError(
                    "a tukey window's two tapered edges must not overlap: need"
                    f" 2 * taper_width <= (k_max - k_min); got taper_width={taper_width},"
                    f" k_min={k_min}, k_max={k_max}.",
                )
            rising_edge = 0.5 * (
                1.0 - numpy.cos(
                    numpy.pi * numpy.clip((k_magnitude - k_min) / taper_width, 0.0, 1.0),
                )
            )
            falling_edge = 0.5 * (
                1.0 - numpy.cos(
                    numpy.pi * numpy.clip((k_max - k_magnitude) / taper_width, 0.0, 1.0),
                )
            )
            in_band = (k_magnitude >= k_min) & (k_magnitude <= k_max)
            return numpy.where(in_band, numpy.minimum(rising_edge, falling_edge), 0.0)
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(
                f"`window` must be 'tophat', 'gaussian', or 'tukey'; got {window!r}.",
            )  # pyright: ignore[reportUnreachable]


##
## === PUBLIC FUNCTIONS
##


def compute_bandpass_filtered_fft(
    *,
    farray: NDArray[Any],
    resolution: tuple[int, ...],
    num_ranks: int,
    k_min: float,
    k_max: float,
    window: FilterWindow = "tophat",
    taper_width: float | None = None,
) -> BandpassFilteredFFT:
    """
    Band-pass filter a field array in Fourier space around [k_min, k_max], returning the
    filtered field alongside the Fourier-space state (before and after masking) that
    produced it. See `_compute_k_mask` for what `window` changes about the filter itself.

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
            "compute_bandpass_filtered_fft expects `farray.ndim == num_ranks + len(resolution)`:"
            f" got farray.ndim={farray.ndim}, num_ranks={num_ranks}, resolution={resolution}.",
        )
    if farray.shape[-num_spatial_dims:] != resolution:
        raise ValueError(
            "compute_bandpass_filtered_fft expects `farray.shape` to end with"
            f" `resolution`: got shape={farray.shape}, resolution={resolution}.",
        )
    _compute_spectra._ensure_isotropic_resolution(
        resolution,
        param_name="compute_bandpass_filtered_fft",
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
    k_mask = _compute_k_mask(
        k_magnitude=k_magnitude,
        k_min=k_min,
        k_max=k_max,
        window=window,
        taper_width=taper_width,
    )
    shifted_fft_farray_masked = shifted_fft_farray * k_mask
    filtered_fft_farray = numpy.fft.ifftshift(
        shifted_fft_farray_masked,
        axes=spatial_axes,
    )
    filtered_farray = numpy.fft.ifftn(
        filtered_fft_farray,
        axes=spatial_axes,
        norm="forward",
    )
    return BandpassFilteredFFT(
        ## `.real` on a complex array is a view, not a copy: left as-is, it would keep the
        ## full complex ifftn buffer (2x the memory) alive for as long as the caller holds it
        filtered_farray=numpy.ascontiguousarray(filtered_farray.real),
        shifted_fft_farray=shifted_fft_farray,
        shifted_fft_farray_masked=shifted_fft_farray_masked,
    )


def compute_bandpass_filtered_farray(
    *,
    farray: NDArray[Any],
    resolution: tuple[int, ...],
    num_ranks: int,
    k_min: float,
    k_max: float,
    window: FilterWindow = "tophat",
    taper_width: float | None = None,
) -> NDArray[Any]:
    """
    Band-pass filter a field array in Fourier space around [k_min, k_max].

    The trailing `len(resolution)` axes are the spatial grid, preceded by `num_ranks`
    leading component axes, e.g. 0 for a scalar, 1 for a vector, 2 for a rank-2 tensor.
    The same FFT kernel serves any rank and any number of spatial dimensions; a component
    axis is never masked separately from any other. See `compute_bandpass_filtered_fft`
    for a caller that also needs the Fourier-space state this was filtered from.
    """
    return compute_bandpass_filtered_fft(
        farray=farray,
        resolution=resolution,
        num_ranks=num_ranks,
        k_min=k_min,
        k_max=k_max,
        window=window,
        taper_width=taper_width,
    ).filtered_farray


## } MODULE
