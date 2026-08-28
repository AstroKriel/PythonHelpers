## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import functools
from dataclasses import dataclass
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_validation import validate_arrays, validate_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class IsotropicPowerSpectrum:
    """Shell-integrated 1D power spectrum."""

    k_bin_centers_1d: NDArray[Any]
    power_spectrum_1d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        validate_arrays.ensure_dims(
            array=self.k_bin_centers_1d,
            param_name="<k_bin_centers_1d>",
            num_dims=1,
        )
        validate_arrays.ensure_dims(
            array=self.power_spectrum_1d,
            param_name="<power_spectrum_1d>",
            num_dims=1,
        )
        if self.k_bin_centers_1d.shape[0] != self.power_spectrum_1d.shape[0]:
            raise ValueError(
                "IsotropicPowerSpectrum arrays must have matching length:"
                f" len(k_bin_centers_1d)={self.k_bin_centers_1d.shape[0]},"
                f" len(power_spectrum_1d)={self.power_spectrum_1d.shape[0]}.",
            )


##
## === INTERNAL HELPERS
##


def _ensure_isotropic_resolution(
    resolution: tuple[int, ...],
    *,
    param_name: str,
) -> None:
    """Ensure every axis in `resolution` has the same number of cells."""
    if not all(num_cells == resolution[0] for num_cells in resolution):
        raise ValueError(
            f"{param_name} assumes an isotropic grid:"
            f" got resolution={resolution} (expected every axis to match).",
        )


@functools.lru_cache(maxsize=10)
def _compute_radial_k_magnitude(
    num_cells_per_dim: tuple[int, ...],
) -> NDArray[Any]:
    """
    Return an N-dimensional scalar array of radial wave-mode indices for an isotropic
    domain with shape `num_cells_per_dim`.

    Each entry stores the index-space distance from the central mode (k=0), so values run
    from ~0 at the center up to k_max near the edges.
    """
    validate_types.ensure_tuple_of_ints(
        param=num_cells_per_dim,
        param_name="<num_cells_per_dim>",
    )
    _ensure_isotropic_resolution(
        num_cells_per_dim,
        param_name="_compute_radial_k_magnitude",
    )
    num_spatial_dims = len(num_cells_per_dim)
    k_center = numpy.array(
        [num_cells // 2 for num_cells in num_cells_per_dim],
        dtype=float,
    ).reshape((num_spatial_dims,) + (1,) * num_spatial_dims)
    grid_indices = numpy.indices(num_cells_per_dim)
    return numpy.linalg.norm(grid_indices - k_center, axis=0)


def _integrate_over_shells(
    *,
    power_spectrum: NDArray[Any],
    resolution: tuple[int, ...],
) -> IsotropicPowerSpectrum:
    """Integrate an N-dimensional power spectrum over radial shells in index-space."""
    validate_arrays.ensure_dims(
        array=power_spectrum,
        param_name="<power_spectrum>",
        num_dims=len(resolution),
    )
    if power_spectrum.shape != resolution:
        raise ValueError(
            "_integrate_over_shells expects `power_spectrum.shape` to match"
            f" `resolution`: got shape={power_spectrum.shape}, resolution={resolution}.",
        )
    _ensure_isotropic_resolution(
        resolution,
        param_name="_integrate_over_shells",
    )
    num_modes = resolution[0] // 2
    k_bin_edges_1d = numpy.linspace(0.5, num_modes, num_modes + 1)
    k_bin_centers_1d = numpy.ceil((k_bin_edges_1d[:-1] + k_bin_edges_1d[1:]) / 2.0)
    k_magnitude = _compute_radial_k_magnitude(num_cells_per_dim=resolution)
    k_bin_mapping = numpy.digitize(
        x=k_magnitude,
        bins=k_bin_edges_1d,
    )
    power_spectrum_1d = numpy.bincount(
        k_bin_mapping.ravel(),
        weights=power_spectrum.ravel(),
        minlength=num_modes + 1,
    )[1:-1]
    return IsotropicPowerSpectrum(
        k_bin_centers_1d=k_bin_centers_1d,
        power_spectrum_1d=power_spectrum_1d,
    )


##
## === PUBLIC FUNCTIONS
##


def compute_power_spectrum_farray(
    *,
    farray: NDArray[Any],
    resolution: tuple[int, ...],
) -> NDArray[Any]:
    """
    Compute the power spectrum of a field array whose trailing `len(resolution)` axes are
    the spatial grid, preceded by zero or more leading component axes, e.g. () for a scalar,
    (3,) for a vector, (3, 3) for a rank-2 tensor.

    Sums |f(k)|^2 over every leading component axis; the same FFT kernel serves any rank and
    any number of spatial dimensions.
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
    num_spatial_dims = len(resolution)
    if farray.shape[-num_spatial_dims:] != resolution:
        raise ValueError(
            "compute_power_spectrum_farray expects `farray.shape` to end with"
            f" `resolution`: got shape={farray.shape}, resolution={resolution}.",
        )
    _ensure_isotropic_resolution(
        resolution,
        param_name="compute_power_spectrum_farray",
    )
    spatial_axes = tuple(range(-num_spatial_dims, 0))
    shifted_fft_farray = numpy.fft.fftshift(
        numpy.fft.fftn(
            farray,
            axes=spatial_axes,
            norm="forward",
        ),
        axes=spatial_axes,
    )
    return numpy.sum(
        numpy.square(
            numpy.abs(
                shifted_fft_farray,
            ),
        ),
        axis=tuple(range(farray.ndim - num_spatial_dims)),
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray: NDArray[Any],
    resolution: tuple[int, ...],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a field array of any rank."""
    power_spectrum = compute_power_spectrum_farray(
        farray=farray,
        resolution=resolution,
    )
    return _integrate_over_shells(
        power_spectrum=power_spectrum,
        resolution=resolution,
    )


## } MODULE
