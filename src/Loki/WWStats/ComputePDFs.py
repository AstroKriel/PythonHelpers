## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import numpy
from Loki.Utils import Utils4Funcs


## ###############################################################
## FUNCTIONS
## ###############################################################
def sampleGaussianDistributionFromQuantiles(p1, p2, x1, x2, num_samples=10**3):
  ## calculate the inverse of the cumulative distribution function (CDF)
  cdf_inv_p1 = numpy.sqrt(2) * numpy.erfinv(2 * p1 - 1)
  cdf_inv_p2 = numpy.sqrt(2) * numpy.erfinv(2 * p2 - 1)
  ## calculate the mean and standard deviation of the normal distribution
  norm_mean = ((x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## generate sampled points from the normal distribution
  samples = norm_mean + norm_std * numpy.random.randn(num_samples)
  return samples

# @Utils4Funcs.time_function
# def computeJPDF(data_x, data_y, bedges_x=None, bedges_y=None, num_bins=None):
#   if (bedges_x is None) and (bedges_y is None) and (num_bins is None):
#     raise ValueError("Error: you did not provide a binning option.")
#   if bedges_x is None: bedges_x = numpy.linspace(numpy.min(data_x), numpy.max(data_x), num_bins+1)
#   if bedges_y is None: bedges_y = numpy.linspace(numpy.min(data_y), numpy.max(data_y), num_bins+1)
#   jpdf, _, _ = numpy.histogram2d(
#     x    = data_x,
#     y    = data_y,
#     bins = [
#       bedges_x,
#       bedges_y
#     ],
#     density = True
#   )
#   return jpdf

@Utils4Funcs.time_function
def computeJPDF(
    data_x              : numpy.ndarray,
    data_y              : numpy.ndarray,
    bedges_x            : numpy.ndarray = None,
    bedges_y            : numpy.ndarray = None,
    num_bins            : int   = None,
    weights             : float = None,
    bedge_extend_factor : float = 0.0,
  ):
  """Compute the 2D joint probability density function (JPDF)."""
  if (bedges_x is None) and (bedges_y is None) and (num_bins is None):
    raise ValueError("Error: you did not provide a binning option.")
  if bedges_x is None: bedges_x = compute1DBins(data_x, num_bins, bedge_extend_factor)
  if bedges_y is None: bedges_y = compute1DBins(data_y, num_bins, bedge_extend_factor)
  bin_counts    = numpy.zeros((len(bedges_x)-1, len(bedges_y)-1), dtype=float)
  bin_indices_x = numpy.clip(numpy.searchsorted(bedges_x, data_x, side="right")-1, 0, len(bedges_x)-2)
  bin_indices_y = numpy.clip(numpy.searchsorted(bedges_y, data_y, side="right")-1, 0, len(bedges_y)-2)
  if weights is None:
    numpy.add.at(bin_counts, (bin_indices_x, bin_indices_y), 1)
  else: numpy.add.at(bin_counts, (bin_indices_x, bin_indices_y), weights)
  bin_area = numpy.abs((bedges_x[1] - bedges_x[0]) * (bedges_y[1] - bedges_y[0]))
  jpdf = bin_counts / (numpy.sum(bin_counts) * bin_area)
  return bedges_x, bedges_y, jpdf

@Utils4Funcs.time_function
def compute1DPDF(
    data     : numpy.ndarray,
    bedges   : numpy.ndarray = None,
    num_bins : int   = None,
    weights  : float = None,
    bedge_extend_factor : float = 0.0
  ):
  """Compute the 1D probability density function (PDF) from the given dataset."""
  if bedges is None:
    if num_bins is None: raise ValueError("Error: you did not provide a binning option.")
    bedges = compute1DBins(data, num_bins, bedge_extend_factor)
  bin_counts = numpy.zeros(len(bedges) - 1, dtype=float)
  bin_width = numpy.abs(bedges[1] - bedges[0])
  bin_indices = numpy.searchsorted(bedges, data, side="right") - 1
  bin_indices = numpy.clip(bin_indices, 0, len(bedges)-2)
  if weights is None:
    numpy.add.at(bin_counts, bin_indices, 1)
  else: numpy.add.at(bin_counts, bin_indices, weights)
  pdf = numpy.append(0, bin_counts / (numpy.sum(bin_counts) * bin_width))
  return bedges, pdf

def compute1DBins(
    data          : numpy.ndarray,
    num_bins      : int,
    extend_factor : float = 0.0
  ):
  data_p16 = numpy.percentile(data, 16)
  data_p50 = numpy.percentile(data, 50)
  data_p84 = numpy.percentile(data, 84)
  return numpy.linspace(
    start = data_p16 - (2 + extend_factor) * (data_p50 - data_p16),
    stop  = data_p84 + (2 + extend_factor) * (data_p84 - data_p50),
    num   = int(num_bins)
  )



## END OF MODULE