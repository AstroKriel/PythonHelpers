## { MODULE

##
## === DEPENDENCIES
##

import numpy
import functools

from dataclasses import dataclass

from jormi.ww_types import type_manager, array_checks, fdata_types, field_types

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class PowerSpectrum1D:
    """Shell-integrated 1D power spectrum."""

    k_bin_centers: numpy.ndarray
    spectrum_1d: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_dims(
            array=self.k_bin_centers,
            param_name="<k_bin_centers>",
            num_dims=1,
        )
        array_checks.ensure_dims(
            array=self.spectrum_1d,
            param_name="<spectrum_1d>",
            num_dims=1,
        )
        if self.k_bin_centers.shape[0] != self.spectrum_1d.shape[0]:
            raise ValueError(
                "PowerSpectrum1D arrays must have matching length:"
                f" len(k_bin_centers)={self.k_bin_centers.shape[0]},"
                f" len(spectrum_1d)={self.spectrum_1d.shape[0]}.",
            )


##
## === INTERNAL HELPERS
##


@functools.lru_cache(maxsize=10)
def _compute_radial_grid(
    grouped_num_cells: tuple[int, ...],
) -> numpy.ndarray:
    """
    Return a grid of radial indices for a 3D cube of shape `grouped_num_cells`.

    The radius is measured in index-space from the central cell.
    """
    type_manager.ensure_sequence(
        param=grouped_num_cells,
        param_name="<grouped_num_cells>",
        seq_length=3,
        valid_elem_types=int,
    )
    k_center = numpy.array(
        [(num_cells - 1) / 2 for num_cells in grouped_num_cells],
        dtype=float,
    )
    k_indices = numpy.indices(grouped_num_cells)
    dx = k_indices[0] - k_center[0]
    dy = k_indices[1] - k_center[1]
    dz = k_indices[2] - k_center[2]
    return numpy.sqrt(dx * dx + dy * dy + dz * dz)


def _compute_3d_power_spectrum(
    sdata: fdata_types.ScalarFieldData | numpy.ndarray,
) -> numpy.ndarray:
    """
    Compute the 3D power spectrum of a scalar field.

    Accepts:
      - ScalarFieldData with shape (Nx, Ny, Nz), or
      - raw 3D ndarray with shape (Nx, Ny, Nz).
    """
    sarray = fdata_types.as_3d_sarray(
        sdata=sdata,
        param_name="<sdata>",
    )
    array_checks.ensure_dims(
        array=sarray,
        param_name="<sdata>",
        num_dims=3,
    )
    fft_field = numpy.fft.fftshift(
        numpy.fft.fftn(
            sarray,
            axes=(0, 1, 2),
            norm="forward",
        ),
        axes=(0, 1, 2),
    )
    power_3d = numpy.square(numpy.abs(fft_field))
    return power_3d


def _compute_spherical_integration(
    spectrum_3d: numpy.ndarray,
) -> PowerSpectrum1D:
    """
    Integrate a 3D power spectrum over spherical shells in index-space.

    Returns a PowerSpectrum1D dataclass.
    """
    array_checks.ensure_dims(
        array=spectrum_3d,
        param_name="<spectrum_3d>",
        num_dims=3,
    )
    num_k_modes = numpy.min(spectrum_3d.shape) // 2
    k_bin_edges = numpy.linspace(0.5, num_k_modes, num_k_modes + 1)
    k_bin_centers = numpy.ceil((k_bin_edges[:-1] + k_bin_edges[1:]) / 2.0)
    grid_k_magn = _compute_radial_grid(spectrum_3d.shape)
    bin_indices = numpy.digitize(
        grid_k_magn,
        k_bin_edges,
    )
    spectrum_1d = numpy.bincount(
        bin_indices.ravel(),
        weights=spectrum_3d.ravel(),
        minlength=num_k_modes + 1,
    )[1:-1]
    return PowerSpectrum1D(
        k_bin_centers=k_bin_centers,
        spectrum_1d=spectrum_1d,
    )


##
## === PUBLIC API
##


def compute_1d_power_spectrum(
    sfield: field_types.ScalarField,
) -> PowerSpectrum1D:
    """
    Compute the 1D (shell-integrated) power spectrum of a ScalarField.

    Parameters
    ----------
    sfield : ScalarField
        3D scalar field defined on a UniformDomain.

    Returns
    -------
    PowerSpectrum1D
        Dataclass with `k_bin_centers` and `spectrum_1d`.
    """
    field_types.ensure_sfield(
        sfield=sfield,
        param_name="<sfield>",
    )
    spectrum_3d = _compute_3d_power_spectrum(
        sdata=sfield.fdata,
    )
    return _compute_spherical_integration(
        spectrum_3d=spectrum_3d,
    )


## } MODULE
