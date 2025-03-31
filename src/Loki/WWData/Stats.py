## START OF MODULE


## ###############################################################
## DEPENDANCIES
## ###############################################################
import numpy
from Loki.WWData import SmoothData


## ###############################################################
## FUNCTIONS
## ###############################################################
def compute_p_norm(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
    p_norm_order: float = 2,
    normalise_by_length: bool = False
  ) -> float:
  """Compute the p_norm_order-norm between two arrays and optionally normalise by num_points^(1/p_norm_order)."""
  array_a = numpy.asarray(array_a)
  array_b = numpy.asarray(array_b)
  errors = []
  if array_a.ndim != 1: errors.append(f"Array-A must be 1-dimensional, but ndim = {array_a.ndim}.")
  if array_b.ndim != 1: errors.append(f"Array-B must be 1-dimensional, but ndim = {array_b.ndim}.")
  if len(array_a) != len(array_b): errors.append(f"Both arrays must have the same number of elems: {len(array_a)} != {len(array_b)}.")
  if array_a.size == 0: errors.append("Array-A should not be empty.")
  if array_b.size == 0: errors.append("Array-B should not be empty.")
  if not isinstance(p_norm_order, (int, float)): errors.append(f"Invalid norm order `p_norm_order = {p_norm_order}`. Must be a number.")
  if errors: raise ValueError("Input validation failed with the following issues:\n" + "\n".join(errors))
  if numpy.all(array_a == array_b): return 0
  array_diff = numpy.abs(array_a - array_b)
  if p_norm_order == numpy.inf: return numpy.max(array_diff)
  elif p_norm_order == 1:
    ## L1 norm: sum of absolute differences
    result = numpy.sum(array_diff)
    if normalise_by_length: result /= len(array_a)
  elif p_norm_order == 0:
    ## L0 pseudo-norm: count of non-zero elements
    result = numpy.count_nonzero(array_diff)
  elif p_norm_order > 0:
    ## general case for p_norm_order > 0
    ## note numerical stability improvement: scale by maximum value
    max_diff = numpy.max(array_diff)
    if max_diff > 0:
        scaled_diff = array_diff / max_diff
        result = max_diff * numpy.power(numpy.sum(numpy.power(scaled_diff, p_norm_order)), 1/p_norm_order)
        if normalise_by_length: result /= numpy.power(len(array_a), 1/p_norm_order)
    else: result = 0
  else: raise ValueError(f"Invalid norm order `p_norm_order={p_norm_order}`. Must be positive or infinity.")
  return result

def sample_gaussian_distribution_from_quantiles(q1, q2, p1, p2, num_samples=10**3):
  """Sample a normal distribution with quantiles 0 < q1 < q2 < 100 and corresponding probabilities 0 < p1 < p2 < 1."""
  if not (0 < q1 < q2 < 1): raise ValueError("Invalid quantile probabilities")
  ## calculate the inverse of the CDF
  cdf_inv_p1 = numpy.sqrt(2) * numpy.erfinv(2 * q1 - 1)
  cdf_inv_p2 = numpy.sqrt(2) * numpy.erfinv(2 * q2 - 1)
  ## calculate the mean and standard deviation of the normal distribution
  norm_mean = ((p1 * cdf_inv_p2) - (p2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
  norm_std = (p2 - p1) / (cdf_inv_p2 - cdf_inv_p1)
  ## generate sampled points from the normal distribution
  samples = norm_mean + norm_std * numpy.random.randn(num_samples)
  return samples

def compute_jpdf(
  data_x         : numpy.ndarray,
  data_y         : numpy.ndarray,
  data_weights   : numpy.ndarray = None,
  bin_edges_cols : numpy.ndarray = None,
  bin_edges_rows : numpy.ndarray = None,
  num_bins       : int = None,
  extend_bin_edge_percent : float = 0.0,
  smoothing_length : float = None,
  ):
  """Compute the 2D joint probability density function (JPDF)."""
  if (len(data_x) == 0) or (len(data_y) == 0):
    raise ValueError("Error: Data arrays must not be empty.")
  if (bin_edges_cols is None) and (bin_edges_rows is None) and (num_bins is None):
    raise ValueError("Error: You did not provide a binning option.")
  if bin_edges_cols is None: bin_edges_cols = compute_1d_bin_edges(data_x, num_bins, extend_bin_edge_percent)
  if bin_edges_rows is None: bin_edges_rows = compute_1d_bin_edges(data_y, num_bins, extend_bin_edge_percent)
  bin_counts = numpy.zeros((len(bin_edges_rows)-1, len(bin_edges_cols)-1), dtype=float)
  bin_indices_cols = numpy.clip(numpy.searchsorted(bin_edges_cols, data_x, side="right")-1, 0, len(bin_edges_cols)-2)
  bin_indices_rows = numpy.clip(numpy.searchsorted(bin_edges_rows, data_y, side="right")-1, 0, len(bin_edges_rows)-2)
  if data_weights is not None:
    if (data_weights.size != data_x.size) or (data_weights.size != data_y.size):
      raise ValueError("Error: The size of `data_weights` must match the size of the provided `data_{x,y}`.")
    numpy.add.at(bin_counts, (bin_indices_rows, bin_indices_cols), data_weights)
  else: numpy.add.at(bin_counts, (bin_indices_rows, bin_indices_cols), 1)
  bin_area = numpy.abs((bin_edges_cols[1] - bin_edges_cols[0]) * (bin_edges_rows[1] - bin_edges_rows[0]))
  jpdf = bin_counts / (numpy.sum(bin_counts) * bin_area)
  if smoothing_length is not None: jpdf = SmoothData.smooth_2d_data_with_gaussian_filter(jpdf, smoothing_length)
  return bin_edges_rows, bin_edges_cols, jpdf

def compute_pdf(
    values    : numpy.ndarray,
    weights   : float = None,
    bin_edges : numpy.ndarray = None,
    num_bins  : int   = None,
    extend_bin_edge_percent : float = 0.0,
    delta_threshold : float = 1e-5,
  ):
  """Compute the 1D probability density function (PDF) from the given dataset."""
  if len(values) == 0: raise ValueError("Error: Cannot compute a PDF for an empty dataset.")
  ## check if the values is essentially a delta function (there is little variantion in values)
  data_std = numpy.std(values)
  if data_std < delta_threshold:
    if values[0] == 0:
      bin_edges = numpy.array([0, 1e-3])
    else: bin_edges = numpy.array([values[0], values[0]*(1+1e-3)])
    bin_width = bin_edges[1] - bin_edges[0]
    pdf = numpy.zeros_like(bin_edges)
    pdf[0] = 1 / bin_width
    return bin_edges, pdf
  if bin_edges is None:
    if num_bins is None: raise ValueError("Error: you did not provide a binning option.")
    bin_edges = compute_1d_bin_edges(values, num_bins, extend_bin_edge_percent)
  elif not numpy.all(bin_edges[:-1] <= bin_edges[1:]):
    raise ValueError("Error: Bin edges must be sorted in ascending order.")
  bin_counts = numpy.zeros(len(bin_edges)-1, dtype=float)
  bin_width  = numpy.abs(bin_edges[1] - bin_edges[0])
  if bin_width == 0: raise ValueError("Error: Bin width is zero.")
  bin_indices = numpy.searchsorted(bin_edges, values, side="right") - 1
  bin_indices = numpy.clip(bin_indices, 0, len(bin_edges)-2)
  if weights is None:
    numpy.add.at(bin_counts, bin_indices, 1)
  else: numpy.add.at(bin_counts, bin_indices, weights)
  total_counts = numpy.sum(bin_counts)
  ## handle when no values points fall into any bins
  if total_counts == 0: return bin_edges, numpy.zeros_like(bin_counts)
  pdf = numpy.append(0, bin_counts / (total_counts * bin_width))
  return bin_edges, pdf

def compute_1d_bin_edges(
    values   : numpy.ndarray,
    num_bins : int,
    extend_bin_edge_percent : float = 0.0
  ):
  data_p16 = numpy.percentile(values, 16)
  data_p50 = numpy.percentile(values, 50)
  data_p84 = numpy.percentile(values, 84)
  return numpy.linspace(
    start = data_p16 - (2 + extend_bin_edge_percent) * (data_p50 - data_p16),
    stop  = data_p84 + (2 + extend_bin_edge_percent) * (data_p84 - data_p50),
    num   = int(num_bins)
  )


## END OF MODULE