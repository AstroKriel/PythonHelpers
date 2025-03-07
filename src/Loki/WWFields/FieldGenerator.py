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
  ## generate white noise in Fourier space
  white_noise = numpy.random.normal(0, 1, (size,)*num_dims)
  ## create a grid of frequencies
  array_k   = numpy.fft.fftfreq(size)
  mg_k_vect = numpy.meshgrid(*(array_k for _ in range(num_dims)), indexing="ij")
  ## compute the magnitude of the wave vector
  mg_k_magnitude = numpy.sqrt(numpy.sum(mg_k_comp**2 for mg_k_comp in mg_k_vect))
  ## create a Gaussian filter in Fourier space
  filter_fft = numpy.exp(-0.5 * (mg_k_magnitude * correlation_length)**2)
  ## apply the filter to the noise in Fourier space
  sfield_fft = filter_fft * numpy.fft.fftn(white_noise)
  ## transform back to real space
  sfield = numpy.real(numpy.fft.ifftn(sfield_fft))
  return sfield


## END OF MODULE