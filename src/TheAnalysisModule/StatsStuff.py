## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np

from TheUsefulModule import WWFuncs


## ###############################################################
## FUNCTIONS
## ###############################################################
def sampleGaussFromQuantiles(p1, p2, x1, x2, num_samples=10**3):
  ## calculate the inverse of the cumulative distribution function (CDF)
  cdf_inv_p1 = np.sqrt(2) * np.erfinv(2 * p1 - 1)
  cdf_inv_p2 = np.sqrt(2) * np.erfinv(2 * p2 - 1)
  ## calculate the mean and standard deviation of the normal distribution
  norm_mean = ((x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## generate sampled points from the normal distribution
  samples = norm_mean + norm_std * np.random.randn(num_samples)
  return samples

@WWFuncs.timeFunc
def compute1DPDF(data, bin_edges=None, num_bins=None, weights=None):
  if bin_edges is not None:
    bin_counts = np.zeros(len(bin_edges)-1, dtype=float)
  elif num_bins is not None:
    bin_edges  = np.linspace(np.min(data), np.max(data), num_bins+1)
    bin_counts = np.zeros(num_bins, dtype=float)
  else: raise ValueError("Error: you did not provide a binning option.")
  ## assume uniform binning
  bin_width = np.abs(bin_edges[1] - bin_edges[0])
  ## use binary search to determine the bin index for each element in the data
  bin_indices = np.searchsorted(bin_edges, data) - 1
  ## increment the corresponding bin count for each element
  if weights is None: np.add.at(bin_counts, bin_indices, 1)
  else: np.add.at(bin_counts, bin_indices, weights)
  ## compute the probability density function
  pdf = np.append(0, bin_counts / np.sum(bin_counts) / bin_width)
  ## return the bin edges and the computed pdf
  return bin_edges, pdf

@WWFuncs.timeFunc
def computeJPDF(data_x, data_y, bedges_x, bedges_y):
  jpdf, _, _ = np.histogram2d(
    x    = data_x,
    y    = data_y,
    bins = [
      bedges_x,
      bedges_y
    ],
    density = True
  )
  return jpdf

@WWFuncs.timeFunc
def compute1DBins(data, num_bins):
  bedges, _ = compute1DPDF(data, num_bins=num_bins)
  median_bedge = np.median(bedges)
  ## extend bins
  bedges = np.linspace(
    median_bedge - 3*np.abs(median_bedge - bedges[0]),
    median_bedge + 3*np.abs(median_bedge - bedges[-1]),
    3*num_bins
  )
  return bedges


## END OF LIBRARY