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
from jormi.ww_arrays._compute_spectra import IsotropicPowerSpectrum as IsotropicPowerSpectrum
from jormi.ww_arrays.farrays_3d import farray_types

##
## === PUBLIC FUNCTIONS
##


def compute_power_spectrum_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    num_ranks: int,
) -> NDArray[Any]:
    """
    Compute the 3D power spectrum of a field array whose trailing 3 axes are the spatial
    grid (num_x0_cells, num_x1_cells, num_x2_cells), preceded by `num_ranks` leading
    component axes, e.g. 0 for a scalar, 1 for a vector, 2 for a rank-2 tensor.
    """
    return _compute_spectra.compute_power_spectrum_farray(
        farray=farray_3d,
        resolution=resolution_3d,
        num_ranks=num_ranks,
    )


def compute_isotropic_power_spectrum_farray(
    *,
    farray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
    num_ranks: int,
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D field array of any rank."""
    return _compute_spectra.compute_isotropic_power_spectrum_farray(
        farray=farray_3d,
        resolution=resolution_3d,
        num_ranks=num_ranks,
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
        num_ranks=0,
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
        num_ranks=1,
    )


def compute_helical_power_spectrum_varray(
    *,
    varray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> NDArray[Any]:
    """
    Compute the signed helical part of the 3D power spectrum of a vector array,
    Re[conj(v(k)) . (i k_hat x v(k))], on the same fftshifted grid as
    `compute_power_spectrum_farray`.

    Each mode is bounded by its power: it equals +|v(k)|^2 for a positive (right-handed)
    helical mode and -|v(k)|^2 for a negative one. For a magnetic field this is k H(k),
    with H(k) the magnetic helicity spectrum.
    """
    farray_types.ensure_3d_varray(
        varray_3d=varray_3d,
        param_name="<varray_3d>",
    )
    spatial_axes = (1, 2, 3)
    shifted_fft_varray = numpy.fft.fftshift(
        numpy.fft.fftn(
            varray_3d,
            axes=spatial_axes,
            norm="forward",
        ),
        axes=spatial_axes,
    )
    k_center = numpy.array(
        [_num_cells // 2 for _num_cells in resolution_3d],
        dtype=float,
    ).reshape((3, 1, 1, 1))
    wavevector_3d = numpy.indices(resolution_3d) - k_center
    k_magnitude = numpy.linalg.norm(wavevector_3d, axis=0)
    unit_wavevector_3d = numpy.divide(
        wavevector_3d,
        k_magnitude,
        out=numpy.zeros_like(wavevector_3d),
        where=(k_magnitude > 0.0),
    )
    rotated_fft_varray = numpy.cross(unit_wavevector_3d, shifted_fft_varray, axis=0)
    return numpy.real(numpy.sum(numpy.conj(shifted_fft_varray) * 1j * rotated_fft_varray, axis=0))


def compute_isotropic_helical_power_spectrum_varray(
    *,
    varray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) signed helical power spectrum of a 3D vector array."""
    helical_power_spectrum = compute_helical_power_spectrum_varray(
        varray_3d=varray_3d,
        resolution_3d=resolution_3d,
    )
    return _compute_spectra._integrate_over_shells(
        power_spectrum=helical_power_spectrum,
        resolution=resolution_3d,
    )


def compute_isotropic_power_spectrum_r2tarray(
    *,
    r2tarray_3d: NDArray[Any],
    resolution_3d: tuple[int, int, int],
) -> IsotropicPowerSpectrum:
    """Compute the 1D (shell-integrated) power spectrum of a 3D rank-2 tensor array."""
    farray_types.ensure_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        param_name="<r2tarray_3d>",
    )
    return compute_isotropic_power_spectrum_farray(
        farray_3d=r2tarray_3d,
        resolution_3d=resolution_3d,
        num_ranks=2,
    )


## } MODULE
