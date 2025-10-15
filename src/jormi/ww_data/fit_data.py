## { MODULE

##
## === DEPENDENCIES
##

import numpy
from typing import Callable
from functools import cached_property
from dataclasses import dataclass
from scipy.optimize import curve_fit as scipy_curve_fit
from jormi.utils import type_utils, array_utils

##
## === DATA TYPES
##


@dataclass(frozen=True)
class DataSeries:
    x_array: numpy.ndarray
    y_array: numpy.ndarray
    x_sigma_array: numpy.ndarray | None = None
    y_sigma_array: numpy.ndarray | None = None

    def __post_init__(
        self,
    ):
        self._validate_data_array(self.x_array)
        self._validate_data_array(self.y_array)
        array_utils.ensure_same_shape(
            array_a=self.x_array,
            array_b=self.y_array,
        )
        if self.x_sigma_array is not None:
            self._validate_sigma_array(
                sigma_array=self.x_sigma_array,
                ref_array=self.x_array,
            )
        if self.y_sigma_array is not None:
            self._validate_sigma_array(
                sigma_array=self.y_sigma_array,
                ref_array=self.y_array,
            )

    @staticmethod
    def _validate_data_array(
        array: numpy.ndarray,
    ):
        array_utils.ensure_nonempty(array)
        array_utils.ensure_finite(array)
        array_utils.ensure_1d(array)
        value_range = numpy.max(array) - numpy.min(array)
        ref_value = numpy.max([
            1.0,
            numpy.median(numpy.abs(array)),
        ])
        if (value_range / ref_value < 1e-2) and (value_range < 1e-9):
            raise ValueError("Data values are (nearly) identical.")

    @staticmethod
    def _validate_sigma_array(
        sigma_array: numpy.ndarray,
        ref_array: numpy.ndarray,
    ):
        array_utils.ensure_nonempty(sigma_array)
        array_utils.ensure_finite(sigma_array)
        array_utils.ensure_1d(sigma_array)
        array_utils.ensure_same_shape(
            array_a=sigma_array,
            array_b=ref_array,
        )
        if numpy.any(sigma_array <= 0):
            raise ValueError("Uncertainty array must be strictly positive.")

    @cached_property
    def num_points(
        self,
    ) -> int:
        return len(self.x_array)

    @cached_property
    def x_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.x_array),
            numpy.max(self.x_array),
        )

    @cached_property
    def y_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.y_array),
            numpy.max(self.y_array),
        )

    @cached_property
    def has_x_uncertainty(
        self,
    ) -> bool:
        return self.x_sigma_array is not None

    @cached_property
    def has_y_uncertainty(
        self,
    ) -> bool:
        return self.y_sigma_array is not None

    def y_weights(
        self
    ) -> numpy.ndarray:
        if self.y_sigma_array is None:
            return numpy.ones_like(self.y_array, dtype=float)
        return 1.0 / numpy.square(self.y_sigma_array)


@dataclass(frozen=True)
class FitStatistic:
    param_name: str
    value: float
    sigma: float | None = None


@dataclass(frozen=True)
class Model:
    model_name: str
    parameter_names: tuple[str, ...]
    function: Callable[..., numpy.ndarray]

    def index_of(
        self,
        parameter_name: str,
    ) -> int:
        return self.parameter_names.index(parameter_name)

    def create_fit_stats(
        self,
        values_vector: list | numpy.ndarray,
        sigmas_vector: list | numpy.ndarray | None = None,
    ) -> dict[str, FitStatistic]:
        values_array = array_utils.as_1d(
            array_like=values_vector,
            check_finite=True,
        )
        if values_array.size != len(self.parameter_names):
            raise ValueError("`values_vector` length does not match `parameter_names`.")
        sigmas_array = None
        if sigmas_vector is not None:
            sigmas_array = array_utils.as_1d(
                array_like=sigmas_vector,
                check_finite=False,
            )
            array_utils.ensure_same_shape(
                array_a=values_array,
                array_b=sigmas_array,
            )
        fit_stats: dict[str, FitStatistic] = {}
        for param_index, param_name in enumerate(self.parameter_names):
            value = float(values_array[param_index])
            sigma = None
            if sigmas_array is not None:
                sigma_value = sigmas_array[param_index]
                sigma = float(sigma_value) if numpy.isfinite(sigma_value) else None
            fit_stats[param_name] = FitStatistic(
                param_name=param_name,
                value=value,
                sigma=sigma,
            )
        return fit_stats


@dataclass(frozen=True)
class FitSummary:
    model: Model
    fit_stats: dict[str, FitStatistic]
    residual_array: numpy.ndarray
    num_points: int
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]

    def __post_init__(self):
        missing_params = set(self.model.parameter_names) - set(self.fit_stats.keys())
        if missing_params:
            missing_str = ", ".join(sorted(missing_params))
            raise ValueError(f"Missing parameters: {missing_str}")
        array_utils.ensure_array(self.residual_array)
        array_utils.ensure_1d(self.residual_array)
        array_utils.ensure_finite(self.residual_array)
        if self.residual_array.size != self.num_points:
            raise ValueError("`num_points` must equal the length of `residual_array`.")

    @cached_property
    def sum_squared_residual(
        self,
    ) -> float:
        return float(numpy.sum(numpy.square(self.residual_array)))

    @cached_property
    def root_mean_square_error(
        self,
    ) -> float:
        return float(numpy.sqrt(numpy.mean(numpy.square(self.residual_array))))

    def get_parameter_values(
        self,
    ) -> list[float]:
        return [self.fit_stats[parameter_name].value for parameter_name in self.model.parameter_names]

    def get_parameter(
        self,
        parameter_name: str,
    ) -> FitStatistic:
        return self.fit_stats[parameter_name]

    @cached_property
    def degrees_of_freedom(self) -> int:
        return self.num_points - len(self.model.parameter_names)

    def evaluate_fit(
        self,
        x_values: list | numpy.ndarray,
    ) -> numpy.ndarray:
        x_array = array_utils.as_1d(
            array_like=x_values,
            check_finite=True,
        )
        return self.model.function(x_array, *self.get_parameter_values())


##
## === FUNCTIONS
##


def fit_linear_model(
    data_series: DataSeries,
) -> FitSummary:
    """Fit a linear model to a 1D data-series using least squares."""
    if data_series.num_points < 3:
        raise ValueError("Need at least 3 points to fit a line.")
    linear_model = Model(
        model_name="linear",
        parameter_names=("intercept", "slope"),
        function=(lambda x_array, intercept, slope: intercept + slope * x_array),
    )
    try:
        fitted_vector, covariance_matrix = scipy_curve_fit(
            f=linear_model.function,
            xdata=data_series.x_array,
            ydata=data_series.y_array,
            sigma=data_series.y_sigma_array if data_series.has_y_uncertainty else None,
            absolute_sigma=True if data_series.has_y_uncertainty else False,
        )
    except RuntimeError as err:
        raise RuntimeError(f"Linear model fit did not converge: {err}") from err
    diag_array = numpy.diag(covariance_matrix)
    stds_vector = numpy.sqrt(diag_array) if numpy.isfinite(diag_array).all() else None
    fit_stats = linear_model.create_fit_stats(
        values_vector=fitted_vector,
        sigmas_vector=stds_vector,  # uncertainties can be ill-conditioned
    )
    residual_array = data_series.y_array - linear_model.function(data_series.x_array, *fitted_vector)
    return FitSummary(
        model=linear_model,
        fit_stats=fit_stats,
        residual_array=residual_array,
        num_points=data_series.num_points,
        x_bounds=data_series.x_bounds,
        y_bounds=data_series.y_bounds,
    )


def fit_line_with_fixed_slope(
    data_series: DataSeries,
    fixed_slope: float,
    y_sigmas: list | numpy.ndarray | None = None,
) -> FitSummary:
    """Fit a line with a fixed slope to a 1D data-series."""
    if data_series.num_points < 2:
        raise ValueError("Need at least 2 points to estimate intercept.")
    if y_sigmas is None:
        weight_array = numpy.ones_like(data_series.y_array, dtype=float)
    else:
        y_sigmas_array = array_utils.as_1d(
            array_like=y_sigmas,
            check_finite=True,
        )
        array_utils.ensure_same_shape(
            array_a=y_sigmas_array,
            array_b=data_series.y_array,
        )
        if numpy.any(y_sigmas_array <= 0) or not numpy.all(numpy.isfinite(y_sigmas_array)):
            raise ValueError("`y_sigmas` must be positive and finite.")
        weight_array = 1.0 / numpy.square(y_sigmas_array)
    x_array = data_series.x_array
    y_array = data_series.y_array
    total_weight = numpy.sum(weight_array)
    y_minus_mx_array = y_array - fixed_slope * x_array
    intercept_value = float(numpy.sum(weight_array * y_minus_mx_array) / total_weight)
    residual_array = y_array - (intercept_value + fixed_slope * x_array)
    ssr_value = float(numpy.sum(weight_array * numpy.square(residual_array)))
    num_dof = data_series.num_points - 1
    sigma_sq = 1.0 if (y_sigmas is not None) else (ssr_value / num_dof)
    intercept_std = float(numpy.sqrt(sigma_sq / total_weight))
    fixed_model = Model(
        model_name="linear_fixed_slope",
        parameter_names=("intercept", "slope"),
        function=(lambda x_array, intercept, slope: intercept + slope * x_array),
    )
    fitted_vector = numpy.array([intercept_value, fixed_slope], dtype=float)
    sigmas_vector = numpy.array([intercept_std, numpy.nan], dtype=float)
    fit_stats = fixed_model.create_fit_stats(
        values_vector=fitted_vector,
        sigmas_vector=sigmas_vector,
    )
    return FitSummary(
        model=fixed_model,
        fit_stats=fit_stats,
        residual_array=residual_array,
        num_points=data_series.num_points,
        x_bounds=data_series.x_bounds,
        y_bounds=data_series.y_bounds,
    )


def get_linear_intercept(
    slope: float,
    x_ref: float,
    y_ref: float,
) -> float:
    """
    Compute the y-intercept (b) for a line y = slope * x + b
    passing through a reference point (x_ref, y_ref).
    """
    if not numpy.isfinite(slope):
        raise ValueError("`slope` must be finite.")
    if not numpy.isfinite(x_ref) or not numpy.isfinite(y_ref):
        raise ValueError("Reference coordinates must be finite.")
    return y_ref - slope * x_ref


def get_powerlaw_coefficient(
    exponent: float,
    x_ref: float,
    y_ref: float,
) -> float:
    """
    Compute the coefficient `A` of a power law:
        `y = A * x^exponent`
    given a reference point `(x_ref, y_ref)`.
    """
    if numpy.isclose(x_ref, 0.0):
        raise ValueError("`x_ref` must be nonzero")
    if (x_ref <= 0.0) and not numpy.isclose(exponent, numpy.round(exponent)):
        raise ValueError("`x_ref` must be positive for non-integer `exponent`")
    return y_ref / x_ref**exponent


def get_line_angle(
    slope: float,
    domain_bounds: tuple[float, float, float, float],
    fig_aspect_ratio: float = 1.0,
) -> float:
    """
    Compute the apparent angle (in degrees) of a line with a particular slope
    when plotted in a rectangular domain fitted on a figure with set aspect ratio.
    """
    type_utils.assert_sequence(
        var_obj=domain_bounds,
        seq_length=4,
        valid_containers=(tuple, list),
        valid_elem_types=(int, float),
        allow_none=False,
    )
    x_min, x_max, y_min, y_max = domain_bounds
    if numpy.isclose(y_max, y_min):
        raise ValueError("`y_min` and `y_max` must not be equal.")
    data_aspect_ratio = (x_max - x_min) / (y_max - y_min)
    scale_x = 1.0
    scale_y = data_aspect_ratio / fig_aspect_ratio
    delta_x = 1.0 * scale_x
    delta_y = slope * scale_y
    angle_rad = numpy.arctan2(delta_y, delta_x)
    angle_deg = angle_rad * 180 / numpy.pi
    return angle_deg


## } MODULE
