## { MODULE

##
## === DEPENDENCIES
##

import numpy
import functools

from dataclasses import dataclass

from jormi.ww_types import array_checks
from jormi.ww_arrays import smooth_2d_arrays

##
## === P-NORM DISTANCE METRIC
##


def compute_p_norm(
    *,
    array_a: numpy.ndarray | list[float],
    array_b: numpy.ndarray | list[float],
    p_norm: float = 2,
    normalise_by_length: bool = False,
) -> float:
    """Compute the p-norm between two arrays and optionally normalise by num_points^(1/p_norm)."""
    array_a = array_checks.as_1d(
        array_like=array_a,
        param_name="array_a",
        check_finite=False,
    )
    array_b = array_checks.as_1d(
        array_like=array_b,
        param_name="array_b",
        check_finite=False,
    )
    mask = numpy.isfinite(array_a) & numpy.isfinite(array_b)
    array_a = array_a[mask]
    array_b = array_b[mask]
    array_checks.ensure_nonempty(
        array=array_a,
        param_name="array_a[finite]",
    )
    array_checks.ensure_same_shape(
        array_a=array_a,
        array_b=array_b,
        param_name_a="array_a[finite]",
        param_name_b="array_b[finite]",
    )
    if not isinstance(p_norm, (int, float, numpy.integer, numpy.floating)):
        raise ValueError(f"`p_norm = {p_norm}` is not a number.")
    if not isinstance(normalise_by_length, (bool, numpy.bool_)):
        raise ValueError("`normalise_by_length` must be a boolean.")
    if numpy.array_equal(array_a, array_b):
        return 0.0
    array_diff = numpy.abs(array_a - array_b)
    if p_norm == numpy.inf:
        return float(numpy.max(array_diff))
    if p_norm == 1:
        value = float(
            numpy.sum(
                array_diff,
                dtype=numpy.float64,
            ),
        )
        if normalise_by_length:
            value /= len(array_a)
        return value
    if p_norm == 0:
        return float(numpy.count_nonzero(array_diff))
    if p_norm > 0:
        max_diff = float(numpy.max(array_diff))
        if max_diff == 0.0:
            return 0.0
        scaled_diff = array_diff / max_diff
        value = max_diff * numpy.power(
            numpy.sum(
                numpy.power(
                    scaled_diff,
                    p_norm,
                    dtype=numpy.float64,
                ),
                dtype=numpy.float64,
            ),
            1.0 / p_norm,
        )
        if normalise_by_length:
            value /= numpy.power(len(array_a), 1.0 / p_norm)
        return float(value)
    raise ValueError(
        f"`p_norm = {p_norm}` is invalid; should either be positive or infinity.",
    )


##
## === BINNING HELPERS
##


def get_bin_edges_from_centers(
    bin_centers: numpy.ndarray,
) -> numpy.ndarray:
    """Convert bin centers to edges."""
    bin_centers = array_checks.as_1d(
        array_like=bin_centers,
        param_name="bin_centers",
        check_finite=True,
    )
    if bin_centers.size < 2:
        raise ValueError("`bin_centers` must contain at least two values.")
    if not numpy.all(bin_centers[1:] >= bin_centers[:-1]):
        raise ValueError("`bin_centers` must be sorted in non-decreasing order.")
    bin_widths = numpy.diff(bin_centers)
    left_bin_edge = bin_centers[0] - 0.5 * bin_widths[0]
    right_bin_edge = bin_centers[-1] + 0.5 * bin_widths[-1]
    return numpy.concatenate(
        [
            numpy.array([left_bin_edge], dtype=numpy.float64),
            bin_centers[:-1] + 0.5 * bin_widths,
            numpy.array([right_bin_edge], dtype=numpy.float64),
        ],
    )


def create_uniformly_spaced_bin_centers(
    *,
    values: numpy.ndarray,
    num_bins: int,
    bin_range_percent: float = 1.0,
) -> numpy.ndarray:
    values = array_checks.as_1d(
        array_like=values,
        param_name="values",
        check_finite=True,
    )
    if not isinstance(num_bins, (int, numpy.integer)):
        raise ValueError("`num_bins` must be an integer.")
    if num_bins <= 0:
        raise ValueError("`num_bins` must be positive.")
    if not isinstance(bin_range_percent, (int, float, numpy.integer, numpy.floating)):
        raise ValueError("`bin_range_percent` must be a number.")
    if not numpy.isfinite(bin_range_percent):
        raise ValueError("`bin_range_percent` must be finite.")
    if bin_range_percent < 0:
        raise ValueError("`bin_range_percent` must be non-negative.")
    p16_value = numpy.percentile(values, 16)
    p50_value = numpy.percentile(values, 50)
    p84_value = numpy.percentile(values, 84)
    return numpy.linspace(
        start=p16_value - (1 + bin_range_percent) * (p50_value - p16_value),
        stop=p84_value + (1 + bin_range_percent) * (p84_value - p50_value),
        num=int(num_bins),
    )


##
## === ESTIMATE 1D PDF
##


@dataclass(frozen=True)
class EstimatedPDF:
    bin_centers: numpy.ndarray
    densities: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_1d(
            array=self.bin_centers,
            param_name="bin_centers",
        )
        array_checks.ensure_nonempty(
            array=self.bin_centers,
            param_name="bin_centers",
        )
        array_checks.ensure_finite(
            array=self.bin_centers,
            param_name="bin_centers",
        )
        array_checks.ensure_1d(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_nonempty(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_finite(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_shape(
            array=self.densities,
            expected_shape=self.bin_centers.shape,
            param_name="densities",
        )

    @functools.cached_property
    def bin_edges(
        self,
    ) -> numpy.ndarray:
        return get_bin_edges_from_centers(self.bin_centers)

    @property
    def resolution(
        self,
    ) -> int:
        return self.bin_centers.shape[0]


def estimate_pdf(
    *,
    values: numpy.ndarray,
    weights: numpy.ndarray | None = None,
    num_bins: int | None = None,
    bin_centers: numpy.ndarray | None = None,
    bin_range_percent: float = 1.0,
    delta_threshold: float = 1e-5,
) -> EstimatedPDF:
    """Compute a 1D probability density function (PDF) for the provided `values`."""
    values = array_checks.as_1d(
        array_like=values,
        param_name="values",
        check_finite=False,
    )
    if weights is not None:
        weights = array_checks.as_1d(
            array_like=weights,
            param_name="weights",
            check_finite=False,
        )
        array_checks.ensure_same_shape(
            array_a=values,
            array_b=weights,
            param_name_a="values",
            param_name_b="weights",
        )
        mask = numpy.isfinite(values) & numpy.isfinite(weights)
        values = values[mask]
        weights = weights[mask]
        array_checks.ensure_nonempty(
            array=values,
            param_name="values[finite]",
        )
        if numpy.any(weights < 0):
            raise ValueError("`weights` must be non-negative.")
    else:
        mask = numpy.isfinite(values)
        values = values[mask]
        array_checks.ensure_nonempty(
            array=values,
            param_name="values[finite]",
        )
        weights = None
    if not isinstance(delta_threshold, (int, float, numpy.integer, numpy.floating)):
        raise ValueError("`delta_threshold` must be a number.")
    if not numpy.isfinite(delta_threshold) or delta_threshold < 0:
        raise ValueError("`delta_threshold` must be a finite, non-negative number.")
    if numpy.std(values, dtype=numpy.float64) < delta_threshold:
        mean_value = float(
            numpy.mean(
                values,
                dtype=numpy.float64,
            ),
        )
        epsilon_width = 1e-4 * (1.0 if mean_value == 0.0 else abs(mean_value))
        bin_centers_delta = numpy.array(
            [
                mean_value - epsilon_width,
                mean_value,
                mean_value + epsilon_width,
            ],
            dtype=numpy.float64,
        )
        bin_edges_delta = numpy.array(
            [
                mean_value - 1.5 * epsilon_width,
                mean_value - 0.5 * epsilon_width,
                mean_value + 0.5 * epsilon_width,
                mean_value + 1.5 * epsilon_width,
            ],
            dtype=numpy.float64,
        )
        bin_widths_delta = numpy.diff(bin_edges_delta)
        bin_counts_delta = numpy.array(
            [0.0, 1.0, 0.0],
            dtype=numpy.float64,
        )
        densities_delta = bin_counts_delta / (
            numpy.sum(
                bin_counts_delta,
                dtype=numpy.float64,
            ) * bin_widths_delta
        )
        return EstimatedPDF(
            bin_centers=bin_centers_delta,
            densities=densities_delta,
        )
    if bin_centers is None:
        if num_bins is None:
            raise ValueError("You did not provide a binning option.")
        bin_centers = create_uniformly_spaced_bin_centers(
            values=values,
            num_bins=num_bins,
            bin_range_percent=bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers = array_checks.as_1d(
            array_like=bin_centers,
            param_name="bin_centers",
            check_finite=True,
        )
        if not numpy.all(bin_centers[:-1] <= bin_centers[1:]):
            raise ValueError("Bin centers must be sorted in ascending order.")
    bin_edges = get_bin_edges_from_centers(bin_centers).astype(numpy.float64)
    bin_widths = numpy.diff(bin_edges)
    if numpy.any(bin_widths <= 0):
        raise ValueError("All bin widths must be positive.")
    bin_indices = numpy.searchsorted(bin_edges, values, side="right") - 1
    bin_indices = numpy.clip(bin_indices, 0, len(bin_centers) - 1)
    bin_counts = numpy.zeros(len(bin_centers), dtype=numpy.float64)
    if weights is not None:
        numpy.add.at(bin_counts, bin_indices, weights)
        total_counts = float(
            numpy.sum(
                weights,
                dtype=numpy.float64,
            ),
        )
    else:
        numpy.add.at(bin_counts, bin_indices, 1.0)
        total_counts = float(
            numpy.sum(
                bin_counts,
                dtype=numpy.float64,
            ),
        )
    if total_counts <= 0:
        raise ValueError("None of the `values` fell into any bins.")
    densities = bin_counts / (total_counts * bin_widths)
    return EstimatedPDF(
        bin_centers=bin_centers,
        densities=densities,
    )


##
## === ESTIMATE 2D JOINT PDF
##


@dataclass(frozen=True)
class EstimatedJPDF:
    bin_centers_rows: numpy.ndarray
    bin_centers_cols: numpy.ndarray
    densities: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_1d(
            array=self.bin_centers_rows,
            param_name="bin_centers_rows",
        )
        array_checks.ensure_nonempty(
            array=self.bin_centers_rows,
            param_name="bin_centers_rows",
        )
        array_checks.ensure_finite(
            array=self.bin_centers_rows,
            param_name="bin_centers_rows",
        )
        array_checks.ensure_1d(
            array=self.bin_centers_cols,
            param_name="bin_centers_cols",
        )
        array_checks.ensure_nonempty(
            array=self.bin_centers_cols,
            param_name="bin_centers_cols",
        )
        array_checks.ensure_finite(
            array=self.bin_centers_cols,
            param_name="bin_centers_cols",
        )
        array_checks.ensure_array(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_dims(
            array=self.densities,
            num_dims=2,
            param_name="densities",
        )
        array_checks.ensure_finite(
            array=self.densities,
            param_name="densities",
        )
        num_rows = self.bin_centers_rows.shape[0]
        num_cols = self.bin_centers_cols.shape[0]
        if self.densities.shape != (num_rows, num_cols):
            raise ValueError(
                "`densities` must have shape "
                f"({num_rows}, {num_cols}); got {self.densities.shape}.",
            )

    @functools.cached_property
    def bin_edges_rows(
        self,
    ) -> numpy.ndarray:
        return get_bin_edges_from_centers(self.bin_centers_rows)

    @functools.cached_property
    def bin_edges_cols(
        self,
    ) -> numpy.ndarray:
        return get_bin_edges_from_centers(self.bin_centers_cols)

    @property
    def resolution(
        self,
    ) -> tuple[int, int]:
        return (
            self.bin_centers_rows.shape[0],
            self.bin_centers_cols.shape[0],
        )


def estimate_jpdf(
    *,
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
    data_x = array_checks.as_1d(
        array_like=data_x,
        param_name="data_x",
        check_finite=False,
    )
    data_y = array_checks.as_1d(
        array_like=data_y,
        param_name="data_y",
        check_finite=False,
    )
    array_checks.ensure_same_shape(
        array_a=data_x,
        array_b=data_y,
        param_name_a="data_x",
        param_name_b="data_y",
    )
    if data_weights is not None:
        data_weights = array_checks.as_1d(
            array_like=data_weights,
            param_name="data_weights",
            check_finite=False,
        )
        array_checks.ensure_same_shape(
            array_a=data_x,
            array_b=data_weights,
            param_name_a="data_x",
            param_name_b="data_weights",
        )
        mask_xy = (numpy.isfinite(data_x) & numpy.isfinite(data_y) & numpy.isfinite(data_weights))
    else:
        mask_xy = numpy.isfinite(data_x) & numpy.isfinite(data_y)
    data_x = data_x[mask_xy]
    data_y = data_y[mask_xy]
    array_checks.ensure_nonempty(
        array=data_x,
        param_name="data_x[finite]",
    )
    if data_weights is not None:
        data_weights = data_weights[mask_xy]
        if numpy.any(data_weights < 0):
            raise ValueError("`data_weights` must be non-negative.")
    else:
        data_weights = None
    if (bin_centers_cols is None) and (bin_centers_rows is None) and (num_bins is None):
        raise ValueError("You did not provide a binning option.")
    if num_bins is None:
        if bin_centers_cols is not None:
            num_bins = len(bin_centers_cols)
        if bin_centers_rows is not None:
            num_bins = len(bin_centers_rows)
    assert num_bins is not None
    if bin_centers_cols is None:
        bin_centers_cols = create_uniformly_spaced_bin_centers(
            values=data_x,
            num_bins=num_bins,
            bin_range_percent=bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers_cols = array_checks.as_1d(
            array_like=bin_centers_cols,
            param_name="bin_centers_cols",
            check_finite=True,
        )
    if bin_centers_rows is None:
        bin_centers_rows = create_uniformly_spaced_bin_centers(
            values=data_y,
            num_bins=num_bins,
            bin_range_percent=bin_range_percent,
        ).astype(numpy.float64)
    else:
        bin_centers_rows = array_checks.as_1d(
            array_like=bin_centers_rows,
            param_name="bin_centers_rows",
            check_finite=True,
        )
    bin_edges_rows = get_bin_edges_from_centers(bin_centers_rows).astype(numpy.float64)
    bin_edges_cols = get_bin_edges_from_centers(bin_centers_cols).astype(numpy.float64)
    bin_widths_rows = numpy.diff(bin_edges_rows)
    bin_widths_cols = numpy.diff(bin_edges_cols)
    if numpy.any(bin_widths_rows <= 0) or numpy.any(bin_widths_cols <= 0):
        raise ValueError("All bin widths must be positive.")
    bin_areas = numpy.outer(bin_widths_rows, bin_widths_cols)
    bin_indices_rows = numpy.searchsorted(bin_edges_rows, data_y, side="right") - 1
    bin_indices_cols = numpy.searchsorted(bin_edges_cols, data_x, side="right") - 1
    bin_indices_rows = numpy.clip(bin_indices_rows, 0, len(bin_centers_rows) - 1)
    bin_indices_cols = numpy.clip(bin_indices_cols, 0, len(bin_centers_cols) - 1)
    bin_counts = numpy.zeros(
        (len(bin_centers_rows), len(bin_centers_cols)),
        dtype=numpy.float64,
    )
    if data_weights is not None:
        numpy.add.at(
            bin_counts,
            (bin_indices_rows, bin_indices_cols),
            data_weights,
        )
    else:
        numpy.add.at(
            bin_counts,
            (bin_indices_rows, bin_indices_cols),
            1.0,
        )
    total_counts = float(
        numpy.sum(
            bin_counts,
            dtype=numpy.float64,
        ),
    )
    estimated_jpdf = (bin_counts / (total_counts * bin_areas) if total_counts > 0 else bin_counts)
    if smoothing_length is not None:
        if not isinstance(
                smoothing_length,
            (int, float, numpy.integer, numpy.floating),
        ):
            raise ValueError("`smoothing_length` must be a number.")
        if not numpy.isfinite(smoothing_length) or smoothing_length < 0:
            raise ValueError(
                "`smoothing_length` must be a finite, non-negative number.",
            )
        estimated_jpdf = smooth_2d_arrays.smooth_2d_data_with_gaussian_filter(
            estimated_jpdf,
            smoothing_length,
        )
        total = float(
            numpy.sum(
                estimated_jpdf * bin_areas,
                dtype=numpy.float64,
            ),
        )
        if total > 0:
            estimated_jpdf /= total
    return EstimatedJPDF(
        bin_centers_rows=bin_centers_rows,
        bin_centers_cols=bin_centers_cols,
        densities=estimated_jpdf,
    )


## } MODULE
