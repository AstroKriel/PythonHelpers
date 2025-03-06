## START OF MODULE


## ###############################################################
## MODULES
## ###############################################################
import numpy
from Loki.WWLogging import FuncUtils


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

@FuncUtils.time_function
def computeJPDF(data_x, data_y, bedges_x, bedges_y):
  jpdf, _, _ = numpy.histogram2d(
    x    = data_x,
    y    = data_y,
    bins = [
      bedges_x,
      bedges_y
    ],
    density = True
  )
  return jpdf

@FuncUtils.time_function
def compute1DPDF(data, bin_edges=None, num_bins=None, weights=None):
  if bin_edges is not None:
    bin_counts = numpy.zeros(len(bin_edges)-1, dtype=float)
  elif num_bins is not None:
    bin_edges  = numpy.linspace(numpy.min(data), numpy.max(data), num_bins+1)
    bin_counts = numpy.zeros(num_bins, dtype=float)
  else: raise ValueError("Error: you did not provide a binning option.")
  ## assume uniform binning
  bin_width = numpy.abs(bin_edges[1] - bin_edges[0])
  ## use binary search to determine the bin index for each element in the data
  bin_indices = numpy.searchsorted(bin_edges, data) - 1
  ## increment the corresponding bin count for each element
  if weights is None: numpy.add.at(bin_counts, bin_indices, 1)
  else: numpy.add.at(bin_counts, bin_indices, weights)
  ## compute the probability density function
  pdf = numpy.append(0, bin_counts / numpy.sum(bin_counts) / bin_width)
  ## return the bin edges and the computed pdf
  return bin_edges, pdf

@FuncUtils.time_function
def compute1DBins(data, num_bins, factor_extend=3):
  bedges, _ = compute1DPDF(data, num_bins=num_bins)
  median_bedge = numpy.median(bedges)
  ## extend bins
  bedges = numpy.linspace(
    start = median_bedge - factor_extend * numpy.abs(median_bedge - bedges[0]),
    stop  = median_bedge + factor_extend * numpy.abs(median_bedge - bedges[-1]),
    num   = factor_extend * num_bins
  )
  return bedges


## END OF MODULE