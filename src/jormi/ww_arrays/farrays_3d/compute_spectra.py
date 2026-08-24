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
from jormi.ww_arrays.farrays_3d import farray_types
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


@functools.lru_cache(maxsize=10)
def _compute_3d_radial_k_magnitude(
    num_cells_per_dim: tuple[int, int, int],
) -> NDArray[Any]:
    """
    Return a 3D scalar array of radial wave-mode indices for a cubic-domain
    with shape `num_cells_per_dim`.

    Each entry stores the index-space distance from the central mode (k=0),
    so values run from ~0 at the center up to k_max near the edges.
    """
    validate_types.ensure_tuple_of_ints(
        param=num_cells_per_dim,
        param_name="<num_cells_per_dim>",
        seq_length=3,
    )
    num_cells_x, num_cells_y, num_cells_z = num_cells_per_dim
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "_compute_3d_radial_k_magnitude assumes a cubic grid:"
            f" got num_cells_per_dim={num_cells_per_dim} (expected num_x0_cells=num_x1_cells=num_x2_cells).",
        )
    k_center = numpy.array(
        [num_cells // 2 for num_cells in num_cells_per_dim],
        dtype=float,
    )
    grid_indices = numpy.indices(num_cells_per_dim)
    delta_ix = grid_indices[0] - k_center[0]
    delta_iy = grid_indices[1] - k_center[1]
    delta_iz = grid_indices[2] - k_center[2]
    return numpy.sqrt(delta_ix * delta_ix + delta_iy * delta_iy + delta_iz * delta_iz)


def _integrate_over_spherical_shells(
    *,
    power_spectrum_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Integrate a 3D power spectrum over spherical shells in index-space."""
    farray_types.ensure_3d_sarray(
        sarray_3d=power_spectrum_3d,
        param_name="<power_spectrum_3d>",
    )
    if power_spectrum_3d.shape != resolution_3d:
        raise ValueError(
            "_integrate_over_spherical_shells expects"
            " `power_spectrum_3d.shape` to match `resolution_3d`:"
            f" got shape={power_spectrum_3d.shape},"
            f" resolution_3d={resolution_3d}.",
        )
    num_cells_x, num_cells_y, num_cells_z = resolution_3d
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "_integrate_over_spherical_shells assumes a cubic grid:"
            f" got resolution_3d={resolution_3d} (expected num_x0_cells=num_x1_cells=num_x2_cells).",
        )
    num_modes = num_cells_x // 2
    k_bin_edges_1d = numpy.linspace(0.5, num_modes, num_modes + 1)
    k_bin_centers_1d = numpy.ceil((k_bin_edges_1d[:-1] + k_bin_edges_1d[1:]) / 2.0)
    k_magn_3d = _compute_3d_radial_k_magnitude(num_cells_per_dim=resolution_3d)
    k_bin_mapping_3d = numpy.digitize(
        x=k_magn_3d,
        bins=k_bin_edges_1d,
    )
    power_spectrum_1d = numpy.bincount(
        k_bin_mapping_3d.ravel(),
        weights=power_spectrum_3d.ravel(),
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
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> NDArray[Any]:
    """
    Compute the 3D power spectrum of a field array whose trailing 3 axes are the spatial
    grid (num_x0_cells, num_x1_cells, num_x2_cells), preceded by zero or more leading
    component axes, e.g. () for a scalar, (3,) for a vector, (3, 3) for a rank-2 tensor.

    Sums |f(k)|^2 over every leading component axis; the same FFT kernel serves any rank.
    """
    validate_arrays.ensure_array(
        array=farray_3d,
        param_name="<farray_3d>",
    )
    validate_types.ensure_tuple_of_ints(
        param=resolution_3d,
        param_name="<resolution_3d>",
        seq_length=3,
        allow_none=False,
    )
    num_cells_x, num_cells_y, num_cells_z = resolution_3d
    if farray_3d.shape[-3:] != resolution_3d:
        raise ValueError(
            "compute_power_spectrum_farray expects `farray_3d.shape[-3:]` to match"
            f" `resolution_3d`: got shape={farray_3d.shape},"
            f" resolution_3d={resolution_3d}.",
        )
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "compute_power_spectrum_farray assumes a cubic grid:"
            f" got resolution_3d={resolution_3d} (expected num_x0_cells=num_x1_cells=num_x2_cells).",
        )
    shifted_fft_farray_3d = numpy.fft.fftshift(
        numpy.fft.fftn(
            farray_3d,
            axes=(-3, -2, -1),
            norm="forward",
        ),
        axes=(-3, -2, -1),
    )
    return numpy.sum(
        numpy.square(
            numpy.abs(
                shifted_fft_farray_3d,
            ),
        ),
        axis=tuple(range(farray_3d.ndim - 3)),
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a field array of any rank."""
    power_spectrum_3d = compute_power_spectrum_farray(
        farray_3d=farray_3d,
        resolution_3d=resolution_3d,
    )
    return _integrate_over_spherical_shells(
        power_spectrum_3d=power_spectrum_3d,
        resolution_3d=resolution_3d,
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
