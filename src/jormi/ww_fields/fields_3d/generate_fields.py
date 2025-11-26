## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
)

##
## === GAUSSIAN RANDOM SARRAYS
##


def _generate_gaussian_random_3d_sarray(
    *,
    resolution: tuple[int, int, int],
    correlation_length: float,
) -> numpy.ndarray:
    """Generate a 3D random scalar array with a Gaussian correlation length."""
    num_cells_x, num_cells_y, num_cells_z = resolution
    sarray_3d_white_noise = numpy.random.normal(
        loc=0.0,
        scale=1.0,
        size=(num_cells_x, num_cells_y, num_cells_z),
    )
    ## cycles per grid length (dimensionless)
    ki_x = numpy.fft.fftfreq(num_cells_x)
    ki_y = numpy.fft.fftfreq(num_cells_y)
    ki_z = numpy.fft.fftfreq(num_cells_z)
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(
        ki_x,
        ki_y,
        ki_z,
        indexing="ij",
    )
    k_magn_grid = numpy.sqrt(
        kx_grid * kx_grid + ky_grid * ky_grid + kz_grid * kz_grid,
    )
    ## Gaussian filter in k-space with lengthscale `correlation_length`
    sarray_3d_fft_filter = numpy.exp(
        -0.5 * numpy.square(k_magn_grid * correlation_length),
    )
    sarray_3d_fft_q = sarray_3d_fft_filter * numpy.fft.fftn(sarray_3d_white_noise)
    sarray_3d_q = numpy.fft.ifftn(sarray_3d_fft_q).real
    return sarray_3d_q


def generate_gaussian_random_3d_sfield(
    *,
    udomain_3d: domain_types.UniformDomain_3D,
    correlation_length: float,
    field_label: str = "G(x)",
    sim_time: float | None = None,
) -> field_types.ScalarField_3D:
    """Generate a 3D scalar field with a Gaussian correlation length."""
    sarray_3d = _generate_gaussian_random_3d_sarray(
        resolution=udomain_3d.resolution,
        correlation_length=correlation_length,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


##
## === POWER-LAW RANDOM SARRAYS
##


def _generate_powerlaw_random_3d_sarray(
    *,
    resolution: tuple[int, int, int],
    alpha_perp: float,
    alpha_para: float | None = None,
) -> numpy.ndarray:
    """
    Generate a 3D random scalar array with a power-law power spectrum.

    If `alpha_para` is None, the amplitude is isotropic:
        |k|^{-alpha_perp}

    If `alpha_para` is provided, the amplitude is anisotropic:
        |k_perp|^{-alpha_perp} |k_parallel|^{-alpha_para}
    """
    num_cells_x, num_cells_y, num_cells_z = resolution
    k_modes_x = numpy.fft.fftfreq(num_cells_x) * num_cells_x
    k_modes_y = numpy.fft.fftfreq(num_cells_y) * num_cells_y
    k_modes_z = numpy.fft.fftfreq(num_cells_z) * num_cells_z
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(
        k_modes_x,
        k_modes_y,
        k_modes_z,
        indexing="ij",
    )
    if alpha_para is None:
        ## isotropic case
        k_magn_grid = numpy.sqrt(
            numpy.square(kx_grid) + numpy.square(ky_grid) + numpy.square(kz_grid),
        )
        k_magn_grid[0, 0, 0] = 1
        sarray_3d_amplitude = numpy.power(
            k_magn_grid,
            -(alpha_perp + 2.0) / 2.0,
        )
    else:
        ## anisotropic case
        k_perp_magn_grid = numpy.sqrt(kx_grid**2 + ky_grid**2)
        k_para_magn_grid = numpy.abs(kz_grid)
        k_perp_magn_grid[k_perp_magn_grid == 0] = 1
        k_para_magn_grid[k_para_magn_grid == 0] = 1
        sarray_3d_amplitude = (
            numpy.power(k_perp_magn_grid, -alpha_perp / 2.0) *
            numpy.power(k_para_magn_grid, -alpha_para / 2.0)
        )
    sarray_3d_random_complex = (
        numpy.random.randn(num_cells_x, num_cells_y, num_cells_z) +
        1j * numpy.random.randn(num_cells_x, num_cells_y, num_cells_z)
    )
    sarray_3d_fft_q = sarray_3d_random_complex * sarray_3d_amplitude
    sarray_3d_q = numpy.fft.ifftn(sarray_3d_fft_q).real
    return sarray_3d_q


def generate_powerlaw_random_3d_sfield(
    *,
    udomain_3d: domain_types.UniformDomain_3D,
    alpha_perp: float,
    alpha_para: float | None = None,
    field_label: str = "P(x)",
    sim_time: float | None = None,
) -> field_types.ScalarField_3D:
    """Generate a 3D scalar field with a power-law power spectrum."""
    sarray_3d = _generate_powerlaw_random_3d_sarray(
        resolution=udomain_3d.resolution,
        alpha_perp=alpha_perp,
        alpha_para=alpha_para,
    )
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        udomain_3d=udomain_3d,
        field_label=field_label,
        sim_time=sim_time,
    )


## } MODULE
