## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import fdata_types, field_types, domain_types

##
## === FUNCTIONS
##


def generate_gaussian_random_sfield(
    *,
    udomain: domain_types.UniformDomain,
    correlation_length: float,
    sim_time: float | None = None,
) -> field_types.ScalarField:
    """Generate a 3D random scalar field with Gaussian correlation length."""
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    num_cells_x, num_cells_y, num_cells_z = udomain.resolution
    white_noise = numpy.random.normal(
        loc=0.0,
        scale=1.0,
        size=(num_cells_x, num_cells_y, num_cells_z),
    )
    ki_x = numpy.fft.fftfreq(num_cells_x)  # cycles per grid length (dimensionless)
    ki_y = numpy.fft.fftfreq(num_cells_y)
    ki_z = numpy.fft.fftfreq(num_cells_z)
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(ki_x, ki_y, ki_z, indexing="ij")
    k_magn_grid = numpy.sqrt(kx_grid * kx_grid + ky_grid * ky_grid + kz_grid * kz_grid)
    ## Gaussian filter in k-space with a particular lengthscale (correlation_length)
    fft_filter = numpy.exp(
        -0.5 * numpy.square(numpy.multiply(k_magn_grid, correlation_length)),
    )
    fft_sarray = fft_filter * numpy.fft.fftn(white_noise)
    sarray = numpy.fft.ifftn(fft_sarray).real
    field_label = (
        rf"$\exp(-(k\,\ell_\mathrm{{cor}})^2/2),"
        rf" \,\ell_\mathrm{{cor}}={correlation_length:.2f}$"
    )
    sdata = fdata_types.ScalarFieldData(
        farray=sarray,
        param_name="<gaussian_random_sfield.fdata>",
    )
    return field_types.ScalarField(
        fdata=sdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


def generate_powerlaw_sfield(
    *,
    udomain: domain_types.UniformDomain,
    alpha_perp: float,
    alpha_para: float | None = None,
    sim_time: float | None = None,
) -> field_types.ScalarField:
    """
    Generate a 3D random scalar field with a power-law power spectrum.

    If `alpha_para` is None, the amplitude is isotropic:
        |k|^{-alpha_perp}

    If `alpha_para` is provided, the amplitude is anisotropic:
        |k_perp|^{-alpha_perp} |k_parallel|^{-alpha_para}
    """
    domain_types.ensure_udomain(
        udomain=udomain,
        param_name="<udomain>",
    )
    num_cells_x, num_cells_y, num_cells_z = udomain.resolution
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
        field_label = rf"$k^{{-{alpha_perp:.2f}}}$"
        k_magn_grid = numpy.sqrt(
            numpy.square(kx_grid) + numpy.square(ky_grid) + numpy.square(kz_grid),
        )
        k_magn_grid[0, 0, 0] = 1
        amplitude = numpy.power(
            k_magn_grid,
            -(alpha_perp + 2.0) / 2.0,
        )
    else:
        ## anisotropic case
        field_label = (rf"$k_\perp^{{-{alpha_perp:.2f}}}"
                       rf" \,k_\parallel^{{-{alpha_para:.2f}}}$")
        k_perp_magn_grid = numpy.sqrt(kx_grid**2 + ky_grid**2)
        k_para_magn_grid = numpy.abs(kz_grid)
        k_perp_magn_grid[k_perp_magn_grid == 0] = 1
        k_para_magn_grid[k_para_magn_grid == 0] = 1
        amplitude = (
            numpy.power(k_perp_magn_grid, -alpha_perp / 2.0) *
            numpy.power(k_para_magn_grid, -alpha_para / 2.0)
        )
    random_sarray = (
        numpy.random.randn(num_cells_x, num_cells_y, num_cells_z) +
        1j * numpy.random.randn(num_cells_x, num_cells_y, num_cells_z)
    )
    fft_spectrum_sarray = random_sarray * amplitude
    spectrum_sarray = numpy.fft.ifftn(fft_spectrum_sarray).real
    sdata = fdata_types.ScalarFieldData(
        farray=spectrum_sarray,
        param_name="<powerlaw_random_sfield.fdata>",
    )
    return field_types.ScalarField(
        fdata=sdata,
        udomain=udomain,
        field_label=field_label,
        sim_time=sim_time,
    )


## } MODULE
