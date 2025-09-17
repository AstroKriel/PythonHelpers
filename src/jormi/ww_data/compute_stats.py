## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass
from jormi.ww_data import smooth_data

##
## === CONTAINERS
##


@dataclass(frozen=True)
class EstimatedPDF:
    bin_centers: numpy.ndarray
    bin_edges: numpy.ndarray
    density: numpy.ndarray


@dataclass(frozen=True)
class EstimatedJPDF:
    bin_centers_rows: numpy.ndarray
    bin_centers_cols: numpy.ndarray
    bin_edges_rows: numpy.ndarray
    bin_edges_cols: numpy.ndarray
    density: numpy.ndarray


##
## === FUNCTIONS
##


def compute_p_norm(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
    p_norm: float = 2,
    normalise_by_length: bool = False,
) -> float:
    """Compute the p-norm between two arrays and optionally normalise by num_points^(1/p_norm)."""
    ## cast inputs to float64 and flatten
    array_a = numpy.asarray(array_a, dtype=numpy.float64).ravel()
    array_b = numpy.asarray(array_b, dtype=numpy.float64).ravel()
    ## mask non-finite values
    mask = numpy.isfinite(array_a) & numpy.isfinite(array_b)
    array_a, array_b = array_a[mask], array_b[mask]
    ## validate inputs
    errors = []
    if array_a.ndim != 1:
        errors.append(f"Array-A must be 1-dimensional, but ndim = {array_a.ndim}.")
    if array_b.ndim != 1:
        errors.append(f"Array-B must be 1-dimensional, but ndim = {array_b.ndim}.")
    if len(array_a) != len(array_b):
        errors.append(f"Both arrays must have the same number of elems: {len(array_a)} != {len(array_b)}.")
    if array_a.size == 0: errors.append("Array-A should not be empty.")
    if array_b.size == 0: errors.append("Array-B should not be empty.")
    if not isinstance(p_norm, (int, float)):
        errors.append(f"`p_norm = {p_norm}` is not a number.")
    if errors: raise ValueError("Validation failed:\n" + "\n".join(errors))
    if numpy.array_equal(array_a, array_b): return 0.0
    array_diff = numpy.abs(array_a - array_b)
    if p_norm == numpy.inf:
        return float(numpy.max(array_diff))
    elif p_norm == 1:
        ## L1 norm: sum of absolute differences (use float64 accumulator)
        value = float(numpy.sum(array_diff, dtype=numpy.float64))
        if normalise_by_length: value /= len(array_a)
        return value
    elif p_norm == 0:
        ## L0 pseudo-norm: count of non-zero elements
        return float(numpy.count_nonzero(array_diff))
    elif p_norm > 0:
        ## general case for p-norm > 0
        ## note: we rescale by the maximum value for numerical stability
        max_diff = float(numpy.max(array_diff))
        if max_diff == 0.0: return 0.0
        scaled_diff = array_diff / max_diff
        value = max_diff * numpy.power(
            numpy.sum(numpy.power(scaled_diff, p_norm, dtype=numpy.float64), dtype=numpy.float64),
            1.0 / p_norm,
        )
        if normalise_by_length: value /= numpy.power(len(array_a), 1.0 / p_norm)
        return float(value)
    else:
        raise ValueError(f"`p_norm = {p_norm}` is invalid. Must be positive or infinity.")


def sample_gaussian_distribution_from_quantiles(q1, q2, p1, p2, num_samples=10**3):
    """Sample a normal distribution with quantiles 0 < q1 < q2 < 100 and corresponding probabilities 0 < p1 < p2 < 1."""
    if not (0 < q1 < q2 < 1): raise ValueError("Invalid quantile probabilities")
    ## inverse CDF
    cdf_inv_p1 = numpy.sqrt(2) * numpy.erfinv(2 * q1 - 1)
    cdf_inv_p2 = numpy.sqrt(2) * numpy.erfinv(2 * q2 - 1)
    ## solve for the mean and standard deviation of the normal distribution
    mean_value = ((p1 * cdf_inv_p2) - (p2 * cdf_inv_p1)) / (cdf_inv_p2 - cdf_inv_p1)
    std_value = (p2 - p1) / (cdf_inv_p2 - cdf_inv_p1)
    ## generate sampled points from the normal distribution
    samples = mean_value + std_value * numpy.random.randn(num_samples)
    return samples


def estimate_pdf(
    values: numpy.ndarray,
    weights: numpy.ndarray | None = None,
    num_bins: int | None = None,
    bin_centers: numpy.ndarray | None = None,
    bin_range_percent: float = 1.0,
    delta_threshold: float = 1e-5,
) -> EstimatedPDF:
    """Compute the 1D probability density function (PDF) for the provided `values`."""
    ## cast to float64, flatten, mask non-finite
    values = numpy.asarray(values, dtype=numpy.float64).ravel()
    if weights is not None:
        weights = numpy.asarray(weights, dtype=numpy.float64).ravel()
        if weights.shape != values.shape: raise ValueError("`weights` length must match `values`.")
        mask = numpy.isfinite(values) & numpy.isfinite(weights)
        values, weights = values[mask], weights[mask]
        if numpy.any(weights < 0): raise ValueError("`weights` must be non-negative.")
    else:
        mask = numpy.isfinite(values)
        values, weights = values[mask], None
    if values.size == 0: raise ValueError("Cannot compute a PDF for an empty dataset.")
    ## degenerate case: very low variance, treat as delta spike
    if numpy.std(values, dtype=numpy.float64) < delta_threshold:
        mean_value = float(numpy.mean(values, dtype=numpy.float64))
        epsilon_width = 1e-4 * (1.0 if mean_value == 0.0 else abs(mean_value))
        bin_centers = numpy.array(
            [
                mean_value - epsilon_width,
                mean_value,
                mean_value + epsilon_width,
            ],
            dtype=numpy.float64,
        )
        bin_edges = numpy.array(
            [
                mean_value - 1.5 * epsilon_width,
                mean_value - 0.5 * epsilon_width,
                mean_value + 0.5 * epsilon_width,
                mean_value + 1.5 * epsilon_width,
            ],
            dtype=numpy.float64,
        )
        bin_widths = numpy.diff(bin_edges)
        bin_counts = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float64)
        density = bin_counts / (numpy.sum(bin_counts, dtype=numpy.float64) * bin_widths)
        return EstimatedPDF(bin_centers=bin_centers, bin_edges=bin_edges, density=density)
    ## determine bin centers
    if bin_centers is None:
        if num_bins is None: raise ValueError("You did not provide a binning option.")
        bin_centers = create_uniformly_spaced_bin_centers(
            values,
            num_bins,
            bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers = numpy.asarray(bin_centers, dtype=numpy.float64).ravel()
        if not numpy.all(bin_centers[:-1] <= bin_centers[1:]):
            raise ValueError("Bin centers must be sorted in ascending order.")
    ## edges from centers
    bin_edges = get_bin_edges_from_centers(bin_centers).astype(numpy.float64)
    bin_widths = numpy.diff(bin_edges)
    if numpy.any(bin_widths <= 0): raise ValueError("All bin widths must be positive.")
    ## assign values to bins
    bin_indices = numpy.searchsorted(bin_edges, values, side="right") - 1
    bin_indices = numpy.clip(bin_indices, 0, len(bin_centers) - 1)
    ## accumulate counts
    bin_counts = numpy.zeros(len(bin_centers), dtype=numpy.float64)
    if weights is not None:
        numpy.add.at(bin_counts, bin_indices, weights)
        total_counts = float(numpy.sum(weights, dtype=numpy.float64))
    else:
        numpy.add.at(bin_counts, bin_indices, 1.0)
        total_counts = float(numpy.sum(bin_counts, dtype=numpy.float64))
    if total_counts <= 0:
        raise ValueError("None of the `values` fell into any bins.")
    ## normalise by bin width and total counts
    density = bin_counts / (total_counts * bin_widths)
    return EstimatedPDF(bin_centers=bin_centers, bin_edges=bin_edges, density=density)


def estimate_jpdf(
    data_x: numpy.ndarray,
    data_y: numpy.ndarray,
    data_weights: numpy.ndarray | None = None,
    bin_centers_cols: numpy.ndarray | None = None,
    bin_centers_rows: numpy.ndarray | None = None,
    num_bins: int | None = None,
    bin_range_percent: float = 1.0,
    smoothing_length: float | None = None,
) -> EstimatedJPDF:
    """Compute the 2D joint probability density function (JPDF)."""
    ## cast to float64, flatten, mask non-finite
    data_x = numpy.asarray(data_x, dtype=numpy.float64).ravel()
    data_y = numpy.asarray(data_y, dtype=numpy.float64).ravel()
    mask_xy = numpy.isfinite(data_x) & numpy.isfinite(data_y)
    data_x, data_y = data_x[mask_xy], data_y[mask_xy]
    if data_x.size == 0 or data_y.size == 0:
        raise ValueError("Data arrays must not be empty.")
    ## handle bin specification
    if (bin_centers_cols is None) and (bin_centers_rows is None) and (num_bins is None):
        raise ValueError("You did not provide a binning option.")
    if num_bins is None:
        if bin_centers_cols is not None: num_bins = len(bin_centers_cols)
        if bin_centers_rows is not None: num_bins = len(bin_centers_rows)
    if bin_centers_cols is None:
        bin_centers_cols = create_uniformly_spaced_bin_centers(
            data_x,
            num_bins,
            bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers_cols = numpy.asarray(bin_centers_cols, dtype=numpy.float64).ravel()
    if bin_centers_rows is None:
        bin_centers_rows = create_uniformly_spaced_bin_centers(
            data_y,
            num_bins,
            bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers_rows = numpy.asarray(bin_centers_rows, dtype=numpy.float64).ravel()
    ## get edges and widths
    bin_edges_rows = get_bin_edges_from_centers(bin_centers_rows).astype(numpy.float64)
    bin_edges_cols = get_bin_edges_from_centers(bin_centers_cols).astype(numpy.float64)
    bin_widths_rows = numpy.diff(bin_edges_rows)
    bin_widths_cols = numpy.diff(bin_edges_cols)
    if numpy.any(bin_widths_rows <= 0) or numpy.any(bin_widths_cols <= 0):
        raise ValueError("All bin widths must be positive.")
    bin_areas = numpy.outer(bin_widths_rows, bin_widths_cols)
    ## map values into bins
    bin_indices_rows = numpy.searchsorted(bin_edges_rows, data_y, side="right") - 1
    bin_indices_cols = numpy.searchsorted(bin_edges_cols, data_x, side="right") - 1
    bin_indices_rows = numpy.clip(bin_indices_rows, 0, len(bin_centers_rows) - 1)
    bin_indices_cols = numpy.clip(bin_indices_cols, 0, len(bin_centers_cols) - 1)
    ## accumulate counts
    bin_counts = numpy.zeros((len(bin_centers_rows), len(bin_centers_cols)), dtype=numpy.float64)
    if data_weights is not None:
        data_weights = numpy.asarray(data_weights, dtype=numpy.float64).ravel()
        if data_weights.shape != data_x.shape:
            raise ValueError(
                "The size of `data_weights` must match the size of `data_{x,y}` after filtering.",
            )
        if numpy.any(~numpy.isfinite(data_weights)) or numpy.any(data_weights < 0):
            raise ValueError("`data_weights` must be finite and non-negative.")
        numpy.add.at(bin_counts, (bin_indices_rows, bin_indices_cols), data_weights)
    else:
        numpy.add.at(bin_counts, (bin_indices_rows, bin_indices_cols), 1.0)
    ## normalise by bin area and total count (float64 accumulator)
    total_counts = float(numpy.sum(bin_counts, dtype=numpy.float64))
    estimated_jpdf = bin_counts / (total_counts * bin_areas) if total_counts > 0 else bin_counts
    ## optional smoothing (then renormalise with cell areas)
    if smoothing_length is not None:
        estimated_jpdf = smooth_data.smooth_2d_data_with_gaussian_filter(estimated_jpdf, smoothing_length)
        total = float(numpy.sum(estimated_jpdf * bin_areas, dtype=numpy.float64))
        if total > 0: estimated_jpdf /= total
    return EstimatedJPDF(
        bin_centers_rows=bin_centers_rows,
        bin_centers_cols=bin_centers_cols,
        bin_edges_rows=bin_edges_rows,
        bin_edges_cols=bin_edges_cols,
        density=estimated_jpdf,
    )


def get_bin_edges_from_centers(bin_centers: numpy.ndarray) -> numpy.ndarray:
    """Convert bin centers to edges."""
    bin_widths = numpy.diff(bin_centers)
    left_bin_edge = bin_centers[0] - 0.5 * bin_widths[0]
    right_bin_edge = bin_centers[-1] + 0.5 * bin_widths[-1]
    return numpy.concatenate([
        [left_bin_edge],
        bin_centers[:-1] + 0.5 * bin_widths,
        [right_bin_edge],
    ])


def create_uniformly_spaced_bin_centers(
    values: numpy.ndarray,
    num_bins: int,
    bin_range_percent: float = 1.0,
):
    p16_value = numpy.percentile(values, 16)
    p50_value = numpy.percentile(values, 50)
    p84_value = numpy.percentile(values, 84)
    return numpy.linspace(
        start=p16_value - (1 + bin_range_percent) * (p50_value - p16_value),
        stop=p84_value + (1 + bin_range_percent) * (p84_value - p50_value),
        num=int(num_bins),
    )


## } MODULE
