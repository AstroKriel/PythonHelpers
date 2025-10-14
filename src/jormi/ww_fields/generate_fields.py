## { MODULE

##
## === DEPENDENCIES
##

import numpy
from jormi.ww_fields import field_types

##
## === FUNCTIONS
##


def generate_gaussian_random_sfield(
    num_cells: int,
    correlation_length: float,
    sim_time: float | None = None,
) -> field_types.ScalarField:
  """Generate a 3D random scalar field with Gaussian correlation length."""
  white_noise = numpy.random.normal(0.0, 1.0, (num_cells, num_cells, num_cells))
  ki_values = numpy.fft.fftfreq(num_cells)  # in cycles per grid length
  kx_grid, ky_grid, kz_grid = numpy.meshgrid(ki_values, ki_values, ki_values, indexing="ij")
  k_magn_grid = numpy.sqrt(kx_grid * kx_grid + ky_grid * ky_grid + kz_grid * kz_grid)
  ## Gaussian filter in k-space with a particular lengthscale (correlation_length)
  fft_filter = numpy.exp(-0.5 * numpy.square(numpy.multiply(k_magn_grid, correlation_length)))
  fft_sarray = fft_filter * numpy.fft.fftn(white_noise)
  sarray = numpy.fft.ifftn(fft_sarray).real
  field_label = rf"$\exp(-(k\,\ell_\mathrm{{cor}})^2/2), \,\ell_\mathrm{{cor}}={correlation_length:.2f}$"
  return field_types.ScalarField(
      sim_time=sim_time,
      data=sarray,
      field_label=field_label,
  )


def generate_powerlaw_sfield(
    num_cells: int,
    alpha_perp: float,
    alpha_para: float | None = None,
    sim_time: float | None = None,
) -> field_types.ScalarField:
    """
    Generates a 3-dimensional random scalar field with a power-law power spectrum:
        - If `alpha_para` is None, the amplitude is isotropic: `|k|^{-alpha_perp}`
        - If `alpha_para` is provided, the amplitude is anisotropic: `|k_perp|^{-alpha_perp} |k_parallel|^{-alpha_para}`
    """
    k_modes = numpy.fft.fftfreq(num_cells) * num_cells
    kx_grid, ky_grid, kz_grid = numpy.meshgrid(k_modes, k_modes, k_modes, indexing="ij")
    if alpha_para is None:
        ## isotropic case
        field_label = rf"$k^{{-{alpha_perp:.2f}}}$"
        k_magn_grid = numpy.sqrt(
            numpy.square(kx_grid) + numpy.square(ky_grid) + numpy.square(kz_grid),
        )
        k_magn_grid[0, 0, 0] = 1
        amplitude = numpy.power(k_magn_grid, -(alpha_perp + 2) / 2.0)
    else:
        ## anisotropic case
        field_label = rf"$k_\perp^{{-{alpha_perp:.2f}}} \,k_\parallel^{{-{alpha_para:.2f}}}$"
        k_perp_magn_grid = numpy.sqrt(kx_grid**2 + ky_grid**2)
        k_para_magn_grid = numpy.abs(kz_grid)
        k_perp_magn_grid[k_perp_magn_grid == 0] = 1
        k_para_magn_grid[k_para_magn_grid == 0] = 1
        amplitude = (
            numpy.power(k_perp_magn_grid, -alpha_perp / 2.0) *
            numpy.power(k_para_magn_grid, -alpha_para / 2.0)
        )
    random_sarray = numpy.random.randn(num_cells, num_cells, num_cells) + 1j * numpy.random.randn(num_cells, num_cells, num_cells)
    fft_spectrum_sarray = random_sarray * amplitude
    spectrum_sarray = numpy.fft.ifftn(fft_spectrum_sarray).real
    return field_types.ScalarField(
      sim_time=sim_time,
      data=spectrum_sarray,
      field_label=field_label,
  )


## } MODULE
