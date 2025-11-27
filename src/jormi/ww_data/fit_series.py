## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Callable
from functools import cached_property
from dataclasses import dataclass
from scipy.optimize import curve_fit as scipy_curve_fit

from jormi.ww_types import type_manager, array_checks
from jormi.ww_data import data_series
from jormi.ww_io import log_manager

##
## === DATA TYPES
##


@dataclass(frozen=True)
class FitStatistic:
    param_name: str
    value: float
    sigma: float | None = None


@dataclass(frozen=True)
class Model:
    model_name: str
    param_names: tuple[str, ...]
    model_fn: Callable[..., numpy.ndarray]

    def index_of(
        self,
        param_name: str,
    ) -> int:
        return self.param_names.index(param_name)

    def create_fit_stats(
        self,
        values_vector: list | numpy.ndarray,
        sigmas_vector: list | numpy.ndarray | None = None,
    ) -> dict[str, FitStatistic]:
        values_array = array_checks.as_1d(
            array_like=values_vector,
            check_finite=True,
        )
        if values_array.size != len(self.param_names):
            raise ValueError("`values_vector` length does not match `param_names`.")
        sigmas_array = None
        if sigmas_vector is not None:
            sigmas_array = array_checks.as_1d(
                array_like=sigmas_vector,
                check_finite=False,
            )
            array_checks.ensure_same_shape(
                array_a=values_array,
                array_b=sigmas_array,
            )
        fit_stats: dict[str, FitStatistic] = {}
        for param_index, param_name in enumerate(self.param_names):
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
        missing_params = set(self.model.param_names) - set(self.fit_stats.keys())
        if missing_params:
            missing_string = ", ".join(sorted(missing_params))
            raise ValueError(f"Missing parameter(s): {missing_string}")
        array_checks.ensure_array(self.residual_array)
        array_checks.ensure_1d(self.residual_array)
        array_checks.ensure_finite(self.residual_array)
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

    def get_param_values(
        self,
    ) -> list[float]:
        return [self.fit_stats[param_name].value for param_name in self.model.param_names]

    def get_param(
        self,
        param_name: str,
    ) -> FitStatistic:
        return self.fit_stats[param_name]

    @cached_property
    def degrees_of_freedom(self) -> int:
        return self.num_points - len(self.model.param_names)

    def evaluate_fit(
        self,
        x_values: list | numpy.ndarray,
    ) -> numpy.ndarray:
        x_data_array = array_checks.as_1d(
            array_like=x_values,
            check_finite=True,
        )
        return self.model.model_fn(x_data_array, *self.get_param_values())


##
## === FUNCTIONS
##


def fit_linear_model(
    data_series: data_series.DataSeries,
) -> FitSummary:
    """Fit a linear model to a 1D data-series using least squares."""
    if data_series.num_points < 3:
        raise ValueError("Need at least 3 points to fit a line.")
    linear_model = Model(
        model_name="linear",
        param_names=("intercept", "slope"),
        model_fn=(lambda x_data_array, intercept, slope: intercept + slope * x_data_array),
    )
    if data_series.x_sigma_array is not None:
        log_manager.log_hint(
            text=(
                "Note: SciPy `curve_fit` does not account for `x_sigma_array` (its ignored); "
                "only `y_sigma_array` is supported in the standard least-squares formalism."
            ),
        )
    try:
        fitted_vector, covariance_matrix = scipy_curve_fit(
            f=linear_model.model_fn,
            xdata=data_series.x_data_array,
            ydata=data_series.y_data_array,
            sigma=data_series.y_sigma_array if data_series.has_y_uncertainty else None,
            absolute_sigma=True if data_series.has_y_uncertainty else False,
        )
    except RuntimeError as err:
        raise RuntimeError(f"Fit failed to converge: {err}") from err
    diag_array = numpy.diag(covariance_matrix)
    sigmas_vector = numpy.sqrt(diag_array) if numpy.isfinite(diag_array).all() else None
    fit_stats = linear_model.create_fit_stats(
        values_vector=fitted_vector,
        sigmas_vector=sigmas_vector,  # uncertainties can be ill-conditioned
    )
    residual_array = data_series.y_data_array - linear_model.model_fn(
        data_series.x_data_array,
        *fitted_vector,
    )
    return FitSummary(
        model=linear_model,
        fit_stats=fit_stats,
        residual_array=residual_array,
        num_points=data_series.num_points,
        x_bounds=data_series.x_bounds,
        y_bounds=data_series.y_bounds,
    )


def fit_line_with_fixed_slope(
    data_series: data_series.DataSeries,
    fixed_slope: float,
) -> FitSummary:
    """Fit a line with a fixed slope to a 1D data-series."""
    if data_series.num_points < 2:
        raise ValueError("Need at least 2 points to estimate intercept.")
    fixed_slope_model = Model(
        model_name="linear_fixed_slope",
        param_names=("intercept", "slope"),
        model_fn=(lambda x_data_array, intercept, _fixed_slope: intercept + _fixed_slope * x_data_array),
    )
    weight_array = data_series.y_weights()
    uses_absolute_sigma = data_series.has_y_uncertainty
    if data_series.has_x_uncertainty:
        log_manager.log_hint(
            text=(
                "Note: `x_sigma_array` is not used in the fixed-slope estimator; "
                "only `y_sigma_array` contributes to weighting."
            ),
        )
    # weighted intercept
    x_data_array = data_series.x_data_array
    y_data_array = data_series.y_data_array
    weight_sum = float(numpy.sum(weight_array))
    y_minus_mx_array = y_data_array - fixed_slope * x_data_array
    intercept_value = float(numpy.sum(weight_array * y_minus_mx_array) / weight_sum)
    # residuals
    residual_array = y_data_array - (intercept_value + fixed_slope * x_data_array)
    # intercept uncertainty
    if uses_absolute_sigma:
        sigma_sq = 1.0
    else:
        num_dof = data_series.num_points - 1
        ssr_value = float(numpy.sum(weight_array * numpy.square(residual_array)))
        sigma_sq = ssr_value / num_dof
    intercept_std = float(numpy.sqrt(sigma_sq / weight_sum))
    fitted_vector = numpy.array([intercept_value, fixed_slope], dtype=float)
    errors_vector = numpy.array([intercept_std, numpy.nan], dtype=float)
    fit_stats = fixed_slope_model.create_fit_stats(
        values_vector=fitted_vector,
        errors_vector=errors_vector, # type: ignore
    )
    return FitSummary(
        model=fixed_slope_model,
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
    when plotted in a rectangular domain stretched over a figure axis with a particular aspect ratio.
    """
    type_manager.ensure_sequence(
        param=domain_bounds,
        seq_length=4,
        valid_seq_types=(tuple, list),
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
