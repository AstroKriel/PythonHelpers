## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def generate_gaussian_random_sfield(
    size               : int,
    correlation_length : float,
    num_dims           : int = 3,
  ) -> numpy.ndarray:
  if num_dims not in [2, 3]: raise ValueError("`num_dims` must be either `2` or `3`.")
  white_noise = numpy.random.normal(0, 1, (size,)*num_dims)
  array_k     = numpy.fft.fftfreq(size)
  grid_k_vec  = numpy.meshgrid(*(array_k for _ in range(num_dims)), indexing="ij")
  grid_k_magn = numpy.sqrt(numpy.sum(grid_k_comp**2 for grid_k_comp in grid_k_vec))
  fft_filter  = numpy.exp(-0.5 * (grid_k_magn * correlation_length)**2)
  sfield_fft  = fft_filter * numpy.fft.fftn(white_noise)
  sfield      = numpy.real(numpy.fft.ifftn(sfield_fft))
  return sfield

def generate_powerlaw_sfield(
    grid_size  : int,
    alpha_perp : float,
    alpha_para : float = None,
  ) -> numpy.ndarray:
  """
  Generates a random scalar field with a power-law power spectrum.
  
  - If `alpha_para` is None, the field will be isotropic, with k^(-alpha_perp).
  - If `alpha_para` is provided, the field will be anisotropic, with k_para^(-alpha_perp) * k_perp^(-alpha_para).
  """
  k_modes = numpy.fft.fftfreq(grid_size) * grid_size
  grid_kx, grid_ky, grid_kz = numpy.meshgrid(k_modes, k_modes, k_modes, indexing="ij")
  if alpha_para is None:
    ## isotropic case
    grid_k_magn = numpy.sqrt(grid_kx**2 + grid_ky**2 + grid_kz**2)
    grid_k_magn[0, 0, 0] = 1
    amplitude = grid_k_magn**(-(alpha_perp + 2) / 2.0)
  else:
    ## anisotropic case
    grid_k_magn_perp = numpy.sqrt(grid_kx**2 + grid_ky**2)
    grid_k_magn_para = numpy.abs(grid_kz)
    grid_k_magn_perp[grid_k_magn_perp == 0] = 1
    grid_k_magn_para[grid_k_magn_para == 0] = 1
    amplitude = grid_k_magn_perp**(-alpha_para / 2.0) * grid_k_magn_para**(-alpha_perp / 2.0)
  random_field = numpy.random.randn(grid_size, grid_size, grid_size) + 1j * numpy.random.randn(grid_size, grid_size, grid_size)
  spectrum_3d = random_field * amplitude
  return numpy.fft.ifftn(spectrum_3d).real


## END OF MODULE