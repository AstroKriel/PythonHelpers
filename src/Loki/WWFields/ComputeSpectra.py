## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def normaliseSpectrum(spectrum: numpy.ndarray) -> numpy.ndarray:
  """Normalizes a power spectrum so that it sums to 1."""
  return spectrum / numpy.sum(spectrum)

def computeAverageNormalisedSpectrum(spectra_group_t: list[numpy.ndarray]) -> numpy.ndarray:
  """Computes the average normalised spectrum over multiple samples."""
  if len(spectra_group_t) == 0: raise ValueError("List of spectra is empty.")
  return numpy.mean(numpy.stack([
    normaliseSpectrum(spectrum)
    for spectrum in spectra_group_t
  ]), axis=0)

def computePowerSpectrum_3D(field: numpy.ndarray) -> numpy.ndarray:
  """Computes the power spectrum of an arbitrary-dimensional field."""
  assert len(field.shape) >= 3, "Field should have at least 3 spatial dimensions."
  fft_field = numpy.fft.fftshift(numpy.fft.fftn(field, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))
  power_spectrum = numpy.sum(
    numpy.power(numpy.abs(fft_field), 2),
    axis = tuple(range(len(field.shape) - 3))
  )
  return power_spectrum

def sphericalIntegrate(spectrum_3d: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
  """Integrates a 3D power spectrum over spherical shells of constant k."""
  grid_kz, grid_ky, grid_kx = numpy.indices(spectrum_3d.shape)
  k_center    = numpy.array([(_ki_size-1)/2.0 for _ki_size in spectrum_3d.shape], dtype=float)
  grid_k_magn = numpy.sqrt((grid_kx - k_center[0])**2 + (grid_ky - k_center[1])**2 + (grid_kz - k_center[2])**2)
  num_k_modes = spectrum_3d.shape[0] // 2
  k_bedges    = numpy.linspace(1, num_k_modes, num_k_modes+1)
  bin_indices = numpy.digitize(grid_k_magn, k_bedges)
  spectrum_1d = numpy.bincount(bin_indices.ravel(), weights=spectrum_3d.ravel(), minlength=num_k_modes+1)[1:]
  k_modes     = (k_bedges[:-1] + k_bedges[1:]) / 2
  return k_modes, spectrum_1d

def computePowerSpectrum_1D(vfield_q: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
  """Computes the full power spectrum including radial integration."""
  spectrum_3d = computePowerSpectrum_3D(vfield_q)
  k_modes, spectrum_1d = sphericalIntegrate(spectrum_3d)
  return k_modes, spectrum_1d


## END OF MODULE