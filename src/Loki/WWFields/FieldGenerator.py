## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def genGaussianRandomField(size: int, correlation_length: float, num_dims:int =3):
  if num_dims not in [2, 3]: raise ValueError("`num_dims` must be either `2` or `3`.")
  white_noise = numpy.random.normal(0, 1, (size,)*num_dims)
  array_k   = numpy.fft.fftfreq(size)
  grid_k_vec = numpy.meshgrid(*(array_k for _ in range(num_dims)), indexing="ij")
  grid_k_magn = numpy.sqrt(numpy.sum(grid_k_comp**2 for grid_k_comp in grid_k_vec))
  filter_fft = numpy.exp(-0.5 * (grid_k_magn * correlation_length)**2)
  sfield_fft = filter_fft * numpy.fft.fftn(white_noise)
  sfield = numpy.real(numpy.fft.ifftn(sfield_fft))
  return sfield


## END OF MODULE