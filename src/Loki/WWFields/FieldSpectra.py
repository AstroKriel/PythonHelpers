## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def normaliseSpectrum(spectrum):
  return numpy.array(spectrum) / numpy.sum(spectrum)

def getAverageNormalisedSpectrum(spectra_group_t):
  return numpy.mean([
    normaliseSpectrum(spectrum)
    for spectrum in spectra_group_t
  ], axis=0)

def compute_power_spectrum_3D(field):
  """
  Computes the power spectrum of a 3D vector field. Note the norm = "forward" 
  in the Fourier transform. This means that the power spectrum will be scaled by
  the number of grid points 1/N^3, and the inverse transform will be scaled by
  N^3.
  """
  # the field should be i,N,N,N
  assert len(field.shape) == 4, "Field should be 3D"
  # Compute the 9 component Fourier Transform and shift zero frequency component to center,
  # then sum all the square components to get the power spectrum   
  return numpy.sum(
    numpy.abs(
      numpy.fft.fftshift(
        numpy.fft.fftn(field, axes=(1,2,3), norm="forward"),
        axes = (1,2,3)
      )
    )**2, axis=0
  )

def compute_tensor_power_spectrum(field: numpy.ndarray) -> numpy.ndarray:
    """
    Computes the power spectrum of a 3D tensor field. Note the norm = "forward" 
    in the Fourier transform. This means that the power spectrum will be scaled by
    the number of grid points 1/N^3, and the inverse transform will be scaled by
    N^3.
    """
    # the field should be 3,3,N,N,N
    assert len(field.shape) == 5, "Field should be a 3D tensor field, 3,3,N,N,N"
    assert field.shape[0] == 3, "Field should be a 3D tensor field, 3,3,N,N,N"
    assert field.shape[1] == 3, "Field should be a 3D tensor field, 3,3,N,N,N"
    # Compute the 9 component Fourier Transform and shift zero frequency component to center,
    # then sum all the square components to get the power spectrum   
    return numpy.sum(
      numpy.abs(
        numpy.fft.fftshift(
          numpy.fft.fftn(field, axes=(2,3,4), norm="forward")
        )
      )**2, axis=(0,1)
    )

def spherical_integrate(data):
    """
    The spherical integrate function takes the 3D power spectrum and integrates
    over spherical shells of constant k. The result is a 1D power spectrum.
    
    It has been tested to reproduce the 1D power spectrum of an input 3D Gaussian
    random field.
    
    It has been tested to maintain the correct normalisation of the power spectrum
    i.e., the integral over the spectrum. For small grids (number of bins) the normalisation
    will be off by a small amount (roughly factor 2 for 128^3 with postive power-law indexes). 
    This is because the frequencies beyond the Nyquist limit are not included in the radial 
    integration. This is not a problem for grids of 256^3 or larger, or for k^-a style spectra,
    which are far more commonly encountered, and the normalisation is closer to 1/10,000 numerical
    error
    
    Args:
        data: The 3D power spectrum
        bins: The number of bins to use for the radial integration. 
              If not specified, the Nyquist limit is used (as should always be the case, anyway).

    Returns:
        k_modes: The k modes corresponding to the radial integration
        radial_sum: The radial integration of the 3D power spectrum (including k^2 correction)
    """
    z, y, x = numpy.indices(data.shape)
    center = numpy.array([
       (i - 1) / 2.0
       for i in data.shape
    ])
    r = numpy.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    bins = data.shape[0] // 2
    bin_edges = numpy.linspace(0.5, bins, bins+1)
    # Use numpy.digitize to assign each element to a bin
    bin_indices = numpy.digitize(r, bin_edges)
    # Compute the radial profile
    radial_sum = numpy.zeros(bins)
    for i in range(1, bins+1):
        mask = bin_indices == i
        radial_sum[i-1] = numpy.sum(data[mask])
    # Generate the spatial frequencies with dk=1
    # Now k_modes represent the bin centers
    k_modes = numpy.ceil((bin_edges[:-1] + bin_edges[1:])/2)
    return k_modes, radial_sum

def computePowerSpectrum(vfield_q):
  spectrum_3d = compute_power_spectrum_3D(vfield_q)
  k_modes, spectrum_1d = spherical_integrate(spectrum_3d)
  return k_modes, spectrum_1d


## END OF MODULE