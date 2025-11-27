## { MODULE

##
## === DEPENDENCIES
##

import numpy
import functools

from dataclasses import dataclass

from jormi.ww_types import array_checks, type_manager
from jormi.ww_fields.fields_3d import (
    _fdata,
    field,
)

##
## === DATA STRUCTURE
##


@dataclass(frozen=True)
class IsotropicPowerSpectrum:
    """Shell-integrated 1D power spectrum."""

    k_bin_centers_1d: numpy.ndarray
    spectrum_1d: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_dims(
            array=self.k_bin_centers_1d,
            param_name="<k_bin_centers_1d>",
            num_dims=1,
        )
        array_checks.ensure_dims(
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
## === INTERNAL HELPERS (ARRAY LEVEL)
##


@functools.lru_cache(maxsize=10)
def _compute_radial_k_magn(
    grouped_num_cells: tuple[int, int, int],
) -> numpy.ndarray:
    """
    Return a 3D scalar array of radial wave-mode indices for a cubic-domain
    with shape `grouped_num_cells`.

    Each entry stores the index-space distance from the central mode (k=0),
    so values run from ~0 at the center up to k_max near the edges.
    """
    type_manager.ensure_tuple_of_ints(
        param=grouped_num_cells,
        param_name="<grouped_num_cells>",
        seq_length=3,
    )
    num_cells_x, num_cells_y, num_cells_z = grouped_num_cells
    if not (num_cells_x == num_cells_y == num_cells_z):
        raise ValueError(
            "_compute_radial_k_magn assumes a cubic grid:"
            f" got grouped_num_cells={grouped_num_cells} (expected Nx=Ny=Nz).",
        )
    k_center = numpy.array(
        [(num_cells - 1) / 2 for num_cells in grouped_num_cells],
        dtype=float,
    )
    grid_indices = numpy.indices(grouped_num_cells)
    delta_ix = grid_indices[0] - k_center[0]
    delta_iy = grid_indices[1] - k_center[1]
    delta_iz = grid_indices[2] - k_center[2]
    return numpy.sqrt(delta_ix * delta_ix + delta_iy * delta_iy + delta_iz * delta_iz)


def _compute_3d_power_spectrum_sarray(
    *,
    sarray_3d_q: numpy.ndarray,
    resolution_3d: tuple[int, int, int],
) -> numpy.ndarray:
    """Compute the 3D power spectrum |F(k)|^2 of a scalar array (Nx, Ny, Nz)."""
    _fdata.ensure_3d_sarray(
        sarray_3d=sarray_3d_q,
        param_name="<sarray_3d_q>",
    )
    type_manager.ensure_tuple_of_ints(
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
            f" got resolution_3d={resolution_3d} (expected Nx=Ny=Nz).",
        )
    sarray_3d_shifted_fft_q = numpy.fft.fftshift(
        numpy.fft.fftn(
            sarray_3d_q,
            axes=(0, 1, 2),
            norm="forward",
        ),
        axes=(0, 1, 2),
    )
    centered_3d_spectrum = numpy.square(numpy.abs(sarray_3d_shifted_fft_q))
    return centered_3d_spectrum


def _integrate_spectrum_over_spherical_shells(
    *,
    centered_3d_spectrum: numpy.ndarray,
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Integrate a 3D power spectrum over spherical shells in index-space."""
    _fdata.ensure_3d_sarray(
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
            f" got resolution_3d={resolution_3d} (expected Nx=Ny=Nz).",
        )
    num_modes = num_cells_x // 2
    k_bin_edges_1d = numpy.linspace(0.5, num_modes, num_modes + 1)
    k_bin_centers_1d = numpy.ceil((k_bin_edges_1d[:-1] + k_bin_edges_1d[1:]) / 2.0)
    k_magn_3d = _compute_radial_k_magn(
        grouped_num_cells=resolution_3d,
    )
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
    sarray_3d: numpy.ndarray,
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
## === PUBLIC FUNCTION
##


def compute_isotropic_power_spectrum_sfield(
    sfield_3d: field.ScalarField_3D,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D scalar field."""
    sarray_3d = field.extract_3d_sarray(sfield_3d)
    udomain_3d = sfield_3d.udomain
    resolution_3d = udomain_3d.resolution
    return _compute_isotropic_power_spectrum_sarray(
        sarray_3d=sarray_3d,
        resolution_3d=resolution_3d,
    )


## } MODULE
