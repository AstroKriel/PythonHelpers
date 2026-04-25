## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import functools

from dataclasses import dataclass

## third-party
import numpy
from typing import Any
from numpy.typing import NDArray

## local
from jormi.ww_fields.fields_3d import (
    _fdata_types,
    field_types,
)
from jormi.ww_checks import check_arrays, check_python_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class IsotropicPowerSpectrum:
    """Shell-integrated 1D power spectrum."""

    k_bin_centers_1d: NDArray[Any]
    spectrum_1d: NDArray[Any]

    def __post_init__(
        self,
    ) -> None:
        check_arrays.ensure_dims(
            array=self.k_bin_centers_1d,
            param_name="<k_bin_centers_1d>",
            num_dims=1,
        )
        check_arrays.ensure_dims(
            array=self.spectrum_1d,
            param_name="<spectrum_1d>",
            num_dims=1,
        )
        if self.k_bin_centers_1d.shape[0] != self.spectrum_1d.shape[0]:
            raise ValueError(
                "IsotropicPowerSpectrum arrays must have matching length:"
                f" len(k_bin_centers_1d)={self.k_bin_centers_1d.shape[0]},"
                f" len(spectrum_1d)={self.spectrum_1d.shape[0]}.",
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
    check_python_types.ensure_tuple_of_ints(
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


def _compute_3d_power_spectrum_sarray(
    *,
    sarray_3d_q: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> NDArray[Any]:
    """Compute the 3D power spectrum |F(k)|^2 of a scalar array (num_x0_cells, num_x1_cells, num_x2_cells)."""
    _fdata_types.ensure_3d_sarray(
        sarray_3d=sarray_3d_q,
        param_name="<sarray_3d_q>",
    )
    check_python_types.ensure_tuple_of_ints(
        param=resolution_3d,
        param_name="<resolution_3d>",
        seq_length=3,
        allow_none=False,
    )
    num_cells_x, num_cells_y, num_cells_z = resolution_3d
    if sarray_3d_q.shape != resolution_3d:
        raise ValueError(
            "_compute_3d_power_spectrum_sarray expects `sarray_3d_q.shape` to match"
            f" `resolution_3d`: got shape={sarray_3d_q.shape},"
            f" resolution_3d={resolution_3d}.",
        )
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "_compute_3d_power_spectrum_sarray assumes a cubic grid:"
            f" got resolution_3d={resolution_3d} (expected num_x0_cells=num_x1_cells=num_x2_cells).",
        )
    sarray_3d_shifted_fft_q = numpy.fft.fftshift(
        numpy.fft.fftn(
            sarray_3d_q,
            axes=(0, 1, 2),
            norm="forward",
        ),
        axes=(0, 1, 2),
    )
    centered_3d_spectrum = numpy.square(
        numpy.abs(
            sarray_3d_shifted_fft_q,
        ),
    )
    return centered_3d_spectrum


def _integrate_spectrum_over_spherical_shells(
    *,
    centered_3d_spectrum: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Integrate a 3D power spectrum over spherical shells in index-space."""
    _fdata_types.ensure_3d_sarray(
        sarray_3d=centered_3d_spectrum,
        param_name="<centered_3d_spectrum>",
    )
    if centered_3d_spectrum.shape != resolution_3d:
        raise ValueError(
            "_integrate_spectrum_over_spherical_shells expects"
            " `centered_3d_spectrum.shape` to match `resolution_3d`:"
            f" got shape={centered_3d_spectrum.shape},"
            f" resolution_3d={resolution_3d}.",
        )
    num_cells_x, num_cells_y, num_cells_z = resolution_3d
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "_integrate_spectrum_over_spherical_shells assumes a cubic grid:"
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
    spectrum_1d = numpy.bincount(
        k_bin_mapping_3d.ravel(),
        weights=centered_3d_spectrum.ravel(),
        minlength=num_modes + 1,
    )[1:-1]
    return IsotropicPowerSpectrum(
        k_bin_centers_1d=k_bin_centers_1d,
        spectrum_1d=spectrum_1d,
    )


def _compute_isotropic_power_spectrum_sarray(
    *,
    sarray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D scalar array."""
    centered_3d_spectrum = _compute_3d_power_spectrum_sarray(
        sarray_3d_q=sarray_3d,
        resolution_3d=resolution_3d,
    )
    return _integrate_spectrum_over_spherical_shells(
        centered_3d_spectrum=centered_3d_spectrum,
        resolution_3d=resolution_3d,
    )


##
## === PUBLIC FUNCTIONS
##


def compute_isotropic_power_spectrum_sfield(
    sfield_3d: field_types.ScalarField_3D,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D scalar field"""
    sarray_3d = field_types.extract_3d_sarray(sfield_3d)
    udomain_3d = sfield_3d.udomain
    resolution_3d = udomain_3d.resolution
    return _compute_isotropic_power_spectrum_sarray(
        sarray_3d=sarray_3d,
        resolution_3d=resolution_3d,
    )


## } MODULE
