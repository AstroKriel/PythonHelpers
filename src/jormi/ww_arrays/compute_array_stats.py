## { MODULE

##
## === DEPENDENCIES
##

import numpy
import functools

from dataclasses import dataclass

from jormi.ww_types import type_manager, array_checks
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
        check_finite=True,
    )
    array_b = array_checks.as_1d(
        array_like=array_b,
        param_name="array_b",
        check_finite=True,
    )
    array_checks.ensure_nonempty(
        array=array_a,
        param_name="array_a",
    )
    array_checks.ensure_same_shape(
        array_a=array_a,
        array_b=array_b,
        param_name_a="array_a",
        param_name_b="array_b",
    )
    type_manager.ensure_numeric(
        param=p_norm,
        param_name="p_norm",
        allow_none=False,
    )
    type_manager.ensure_bool(
        param=normalise_by_length,
        param_name="normalise_by_length",
        allow_none=False,
    )
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


def _create_uniformly_spaced_bin_centers(
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
    type_manager.ensure_finite_int(
        param=num_bins,
        param_name="num_bins",
        allow_none=False,
        require_positive=True,
    )
    type_manager.ensure_finite_float(
        param=bin_range_percent,
        param_name="bin_range_percent",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    p16_value = numpy.percentile(values, 16)
    p50_value = numpy.percentile(values, 50)
    p84_value = numpy.percentile(values, 84)
    start_value = p16_value - (1 + bin_range_percent) * (p50_value - p16_value)
    stop_value = p84_value + (1 + bin_range_percent) * (p84_value - p50_value)
    return numpy.linspace(start_value, stop_value, num_bins)


def _get_bin_edges_from_centers(
    bin_centers: numpy.ndarray,
) -> numpy.ndarray:
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


##
## === PDF NORMALISATION CHECKS
##


def _ensure_correct_pdf_integral(
    *,
    bin_centers: numpy.ndarray,
    densities: numpy.ndarray,
    param_name: str = "<EstimatedPDF>",
    relative_tol: float = 1e-6,
    absolute_tol: float = 1e-12,
) -> None:
    type_manager.ensure_finite_float(
        param=relative_tol,
        param_name="relative_tol",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_manager.ensure_finite_float(
        param=absolute_tol,
        param_name="absolute_tol",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    bin_edges = _get_bin_edges_from_centers(bin_centers)
    bin_widths = numpy.diff(bin_edges)
    integral = float(
        numpy.sum(
            densities * bin_widths,
            dtype=numpy.float64,
        ),
    )
    type_manager.ensure_finite_float(
        param=integral,
        param_name="integral",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    if not numpy.isclose(integral, 1.0, rtol=relative_tol, atol=absolute_tol):
        raise ValueError(f"{param_name} is not normalised: integral={integral}.")


def _ensure_correct_jpdf_integral(
    *,
    row_centers: numpy.ndarray,
    col_centers: numpy.ndarray,
    densities: numpy.ndarray,
    param_name: str = "<EstimatedJPDF>",
    relative_tol: float = 1e-6,
    absolute_tol: float = 1e-12,
) -> None:
    type_manager.ensure_finite_float(
        param=relative_tol,
        param_name="relative_tol",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    type_manager.ensure_finite_float(
        param=absolute_tol,
        param_name="absolute_tol",
        allow_none=False,
        require_positive=True,
        allow_zero=True,
    )
    row_edges = _get_bin_edges_from_centers(row_centers)
    col_edges = _get_bin_edges_from_centers(col_centers)
    bin_widths_rows = numpy.diff(row_edges)
    bin_widths_cols = numpy.diff(col_edges)
    bin_areas = numpy.outer(bin_widths_rows, bin_widths_cols)
    integral = float(
        numpy.sum(
            densities * bin_areas,
            dtype=numpy.float64,
        ),
    )
    type_manager.ensure_finite_float(
        param=integral,
        param_name="integral",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
    if not numpy.isclose(integral, 1.0, rtol=relative_tol, atol=absolute_tol):
        raise ValueError(f"{param_name} is not normalised: integral={integral}.")


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
        array_checks.ensure_finite(
            array=self.bin_centers,
            param_name="bin_centers",
        )
        array_checks.ensure_1d(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_finite(
            array=self.densities,
            param_name="densities",
        )
        array_checks.ensure_same_shape(
            array_a=self.bin_centers,
            array_b=self.densities,
            param_name_a="bin_centers",
            param_name_b="densities",
        )
        _ensure_correct_pdf_integral(
            bin_centers=self.bin_centers,
            densities=self.densities,
            param_name="EstimatedPDF",
        )

    @functools.cached_property
    def bin_edges(
        self,
    ) -> numpy.ndarray:
        return _get_bin_edges_from_centers(self.bin_centers)

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
    type_manager.ensure_finite_float(
        param=delta_threshold,
        param_name="delta_threshold",
        allow_none=False,
        require_positive=True,
        allow_zero=False,
    )
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
            bin_widths_delta * numpy.sum(
                bin_counts_delta,
                dtype=numpy.float64,
            )
        )
        return EstimatedPDF(
            bin_centers=bin_centers_delta,
            densities=densities_delta,
        )
    if bin_centers is None:
        if num_bins is None:
            raise ValueError("You did not provide a binning option.")
        bin_centers = _create_uniformly_spaced_bin_centers(
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
    bin_edges = _get_bin_edges_from_centers(bin_centers).astype(numpy.float64)
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
    row_centers: numpy.ndarray
    col_centers: numpy.ndarray
    densities: numpy.ndarray

    def __post_init__(
        self,
    ) -> None:
        array_checks.ensure_1d(
            array=self.row_centers,
            param_name="row_centers",
        )
        array_checks.ensure_finite(
            array=self.row_centers,
            param_name="row_centers",
        )
        array_checks.ensure_1d(
            array=self.col_centers,
            param_name="col_centers",
        )
        array_checks.ensure_finite(
            array=self.col_centers,
            param_name="col_centers",
        )
        array_checks.ensure_dims(
            array=self.densities,
            param_name="densities",
            num_dims=2,
        )
        array_checks.ensure_finite(
            array=self.densities,
            param_name="densities",
        )
        num_rows = self.row_centers.shape[0]
        num_cols = self.col_centers.shape[0]
        array_checks.ensure_shape(
            array=self.densities,
            param_name="densities",
            expected_shape=(num_rows, num_cols),
        )
        _ensure_correct_jpdf_integral(
            row_centers=self.row_centers,
            col_centers=self.col_centers,
            densities=self.densities,
            param_name="EstimatedJPDF",
        )

    @functools.cached_property
    def row_edges(
        self,
    ) -> numpy.ndarray:
        return _get_bin_edges_from_centers(self.row_centers)

    @functools.cached_property
    def col_edges(
        self,
    ) -> numpy.ndarray:
        return _get_bin_edges_from_centers(self.col_centers)

    @property
    def resolution(
        self,
    ) -> tuple[int, int]:
        return (
            self.row_centers.shape[0],
            self.col_centers.shape[0],
        )


def estimate_jpdf(
    *,
    data_x: numpy.ndarray,
    data_y: numpy.ndarray,
    data_weights: numpy.ndarray | None = None,
    col_centers: numpy.ndarray | None = None,
    row_centers: numpy.ndarray | None = None,
    num_bins: int | None = None,
    bin_range_percent: float = 1.0,
    smoothing_length: float | None = None,
) -> EstimatedJPDF:
    """Compute the 2D joint probability density function (JPDF)."""
    data_x = array_checks.as_1d(
        array_like=data_x,
        param_name="data_x",
        check_finite=True,
    )
    data_y = array_checks.as_1d(
        array_like=data_y,
        param_name="data_y",
        check_finite=True,
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
            check_finite=True,
        )
        array_checks.ensure_same_shape(
            array_a=data_x,
            array_b=data_weights,
            param_name_a="data_x",
            param_name_b="data_weights",
        )
        if numpy.any(data_weights < 0):
            raise ValueError("`data_weights` must be non-negative.")
    else:
        data_weights = None
    if (col_centers is None) and (row_centers is None) and (num_bins is None):
        raise ValueError("You did not provide a binning option.")
    if num_bins is None:
        if col_centers is not None:
            num_bins = len(col_centers)
        if row_centers is not None:
            num_bins = len(row_centers)
    assert num_bins is not None
    if col_centers is None:
        col_centers = _create_uniformly_spaced_bin_centers(
            values=data_x,
            num_bins=num_bins,
            bin_range_percent=bin_range_percent,
        ).astype(numpy.float64)
    else:
        col_centers = array_checks.as_1d(
            array_like=col_centers,
            param_name="col_centers",
            check_finite=True,
        )
    if row_centers is None:
        row_centers = _create_uniformly_spaced_bin_centers(
            values=data_y,
            num_bins=num_bins,
            bin_range_percent=bin_range_percent,
        ).astype(numpy.float64)
    else:
        row_centers = array_checks.as_1d(
            array_like=row_centers,
            param_name="row_centers",
            check_finite=True,
        )
    row_edges = _get_bin_edges_from_centers(row_centers).astype(numpy.float64)
    col_edges = _get_bin_edges_from_centers(col_centers).astype(numpy.float64)
    bin_widths_rows = numpy.diff(row_edges)
    bin_widths_cols = numpy.diff(col_edges)
    if numpy.any(bin_widths_rows <= 0) or numpy.any(bin_widths_cols <= 0):
        raise ValueError("All bin widths must be positive.")
    bin_areas = numpy.outer(bin_widths_rows, bin_widths_cols)
    bin_indices_rows = numpy.searchsorted(row_edges, data_y, side="right") - 1
    bin_indices_cols = numpy.searchsorted(col_edges, data_x, side="right") - 1
    bin_indices_rows = numpy.clip(bin_indices_rows, 0, len(row_centers) - 1)
    bin_indices_cols = numpy.clip(bin_indices_cols, 0, len(col_centers) - 1)
    bin_counts = numpy.zeros(
        (
            len(row_centers),
            len(col_centers),
        ),
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
        type_manager.ensure_finite_float(
            param=smoothing_length,
            param_name="smoothing_length",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )
        estimated_jpdf = smooth_2d_arrays.smooth_2d_array(
            data=estimated_jpdf,
            sigma=smoothing_length,
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
        row_centers=row_centers,
        col_centers=col_centers,
        densities=estimated_jpdf,
    )


## } MODULE
