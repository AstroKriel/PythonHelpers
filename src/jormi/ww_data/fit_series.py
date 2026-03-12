## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Callable
from functools import cached_property
from dataclasses import dataclass
from scipy.optimize import curve_fit as scipy_curve_fit

from jormi.ww_io import log_manager
from jormi.utils import list_utils
from jormi.ww_types import type_checks, array_checks
from jormi.ww_arrays import compute_array_stats
from jormi.ww_data.series import GaussianSeries

##
## === UTILITY FUNCTIONS
##


def get_linear_intercept(
    slope: float,
    x_ref: float,
    y_ref: float,
) -> float:
    """
    Compute the y-intercept (b) for a line y = slope * x + b
    passing through a reference point (x_ref, y_ref).
    """
    type_checks.ensure_finite_float(
        param=slope,
        param_name="slope",
    )
    type_checks.ensure_finite_float(
        param=x_ref,
        param_name="x_ref",
    )
    type_checks.ensure_finite_float(
        param=y_ref,
        param_name="y_ref",
    )
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
    type_checks.ensure_finite_float(
        param=exponent,
        param_name="exponent",
    )
    type_checks.ensure_finite_float(
        param=x_ref,
        param_name="x_ref",
    )
    type_checks.ensure_finite_float(
        param=y_ref,
        param_name="y_ref",
    )
    if numpy.isclose(x_ref, 0.0):
        raise ValueError("`x_ref` must be nonzero.")
    if (x_ref <= 0.0) and not numpy.isclose(exponent, numpy.round(exponent)):
        raise ValueError("`x_ref` must be positive for non-integer `exponent`.")
    return y_ref / x_ref**exponent


def get_line_angle(
    slope: float,
    domain_bounds: tuple[float, float, float, float],
    aspect_ratio: float = 1.0,
) -> float:
    """
    Compute the apparent angle (in degrees) of a line with a particular slope
    when plotted in a rectangular domain stretched to have a particular aspect ratio.
    """
    ## validate scalars
    type_checks.ensure_finite_float(
        param=slope,
        param_name="slope",
    )
    type_checks.ensure_finite_float(
        param=aspect_ratio,
        param_name="aspect_ratio",
    )
    if aspect_ratio <= 0.0:
        raise ValueError("`aspect_ratio` must be positive.")
    ## validate domain_bounds
    type_checks.ensure_sequence(
        param=domain_bounds,
        seq_length=4,
        valid_seq_types=(tuple, list),
        valid_elem_types=(int, float),
        allow_none=False,
    )
    x_min, x_max, y_min, y_max = domain_bounds
    if numpy.isclose(x_max, x_min):
        raise ValueError("`x_min` and `x_max` must not be equal.")
    if numpy.isclose(y_max, y_min):
        raise ValueError("`y_min` and `y_max` must not be equal.")
    ## compute angle
    data_aspect_ratio = (x_max - x_min) / (y_max - y_min)
    scale_y = data_aspect_ratio / aspect_ratio
    angle_rad = numpy.arctan2(slope * scale_y, 1.0)
    angle_deg = angle_rad * 180 / numpy.pi
    return angle_deg


##
## === FIT STATISTIC CLASS
##


@dataclass(frozen=True)
class FitStatistic:
    """Fitted value and optional uncertainty for a single model parameter."""

    param_name: str
    value: float
    sigma: float | None = None


##
## === MODEL CLASS
##


@dataclass(frozen=True)
class Model:
    """A named model: function, parameter names, and an index lookup helper."""

    model_name: str
    model_fn: Callable[..., numpy.ndarray]
    param_names: tuple[str, ...]

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


##
## === FIT SUMMARY CLASS
##


@dataclass(frozen=True)
class FitSummary:
    """
    Complete result of a curve-fit: model, per-parameter statistics, and residuals.

    Fields
    ---
    - `model`:
        The fitted model (name, parameter names, model function).

    - `fit_stats`:
        Mapping from parameter name to its `FitStatistic` (value + optional sigma).

    - `residual_array`:
        1D array of (y_data - y_fit) residuals; length must equal `num_points`.

    - `num_points`:
        Number of data points used in the fit.

    - `x_bounds`:
        (min, max) of the x data used in the fit.

    - `y_bounds`:
        (min, max) of the y data used in the fit.
    """

    model: Model
    fit_stats: dict[str, FitStatistic]
    residual_array: numpy.ndarray
    num_points: int
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]

    def __post_init__(self):
        missing_params = set(self.model.param_names) - set(self.fit_stats.keys())
        if missing_params:
            missing_string = list_utils.as_string(
                elems=sorted(missing_params),
                wrap_in_quotes=True,
                conjunction="and",
            )
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
        return float(
            numpy.sum(
                numpy.square(
                    self.residual_array,
                ),
            ),
        )

    @cached_property
    def rms_error(
        self,
    ) -> float:
        return compute_array_stats.compute_rms(self.residual_array)

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
## === FIT FUNCTIONS
##


def fit_linear_model(
    series: GaussianSeries,
) -> FitSummary:
    """Fit a linear model to a 1D data-series using least squares."""
    type_checks.ensure_type(
        param=series,
        valid_types=GaussianSeries,
        param_name="series",
    )
    if series.num_points < 3:
        raise ValueError("Need at least 3 points to fit a line.")
    linear_model = Model(
        model_name="linear",
        param_names=("intercept", "slope"),
        model_fn=(lambda x_data_array, intercept, slope: intercept + slope * x_data_array),
    )
    if series.x_sigmas is not None:
        log_manager.log_hint(
            text=(
                "Note: SciPy `curve_fit` does not account for `x_sigmas` (it is ignored); "
                "only `y_sigmas` is supported in the standard least-squares formalism."
            ),
        )
    try:
        fitted_vector, covariance_matrix = scipy_curve_fit(
            f=linear_model.model_fn,
            xdata=series.x_values,
            ydata=series.y_values,
            sigma=series.y_sigmas,
            absolute_sigma=series.y_sigmas is not None,
        )
    except RuntimeError as err:
        raise RuntimeError(f"Fit failed to converge: {err}") from err
    diag_array = numpy.diag(covariance_matrix)
    sigmas_vector = numpy.sqrt(diag_array) if numpy.isfinite(diag_array).all() else None
    fit_stats = linear_model.create_fit_stats(
        values_vector=fitted_vector,
        sigmas_vector=sigmas_vector,  # uncertainties can be ill-conditioned
    )
    residual_array = series.y_values - linear_model.model_fn(
        series.x_values,
        *fitted_vector,
    )
    return FitSummary(
        model=linear_model,
        fit_stats=fit_stats,
        residual_array=residual_array,
        num_points=series.num_points,
        x_bounds=series.x_bounds,
        y_bounds=series.y_bounds,
    )


def fit_line_with_fixed_slope(
    series: GaussianSeries,
    fixed_slope: float,
) -> FitSummary:
    """Fit a line with a fixed slope to a 1D data-series."""
    type_checks.ensure_type(
        param=series,
        valid_types=GaussianSeries,
        param_name="series",
    )
    type_checks.ensure_finite_float(
        param=fixed_slope,
        param_name="fixed_slope",
    )
    if series.num_points < 2:
        raise ValueError("Need at least 2 points to estimate intercept.")
    fixed_slope_model = Model(
        model_name="linear_fixed_slope",
        param_names=("intercept", "slope"),
        model_fn=(lambda x_data_array, intercept, _fixed_slope: intercept + _fixed_slope * x_data_array),
    )
    weight_array = series.y_weights()
    uses_absolute_sigma = series.y_sigmas is not None
    if series.x_sigmas is not None:
        log_manager.log_hint(
            text=(
                "Note: `x_sigmas` is not used in the fixed-slope estimator; "
                "only `y_sigmas` contributes to weighting."
            ),
        )
    ## weighted intercept
    x_values = series.x_values
    y_values = series.y_values
    weight_sum = float(numpy.sum(weight_array))
    y_minus_mx_array = y_values - fixed_slope * x_values
    intercept_value = float(numpy.sum(weight_array * y_minus_mx_array) / weight_sum)
    ## residuals
    residual_array = y_values - (intercept_value + fixed_slope * x_values)
    ## intercept uncertainty
    if uses_absolute_sigma:
        sigma_sq = 1.0
    else:
        num_dof = series.num_points - 1
        ssr_value = float(numpy.sum(weight_array * numpy.square(residual_array)))
        sigma_sq = ssr_value / num_dof
    intercept_std = float(numpy.sqrt(sigma_sq / weight_sum))
    fitted_vector = numpy.array([intercept_value, fixed_slope], dtype=float)
    sigmas_vector = numpy.array([intercept_std, numpy.nan], dtype=float)
    fit_stats = fixed_slope_model.create_fit_stats(
        values_vector=fitted_vector,
        sigmas_vector=sigmas_vector,
    )
    return FitSummary(
        model=fixed_slope_model,
        fit_stats=fit_stats,
        residual_array=residual_array,
        num_points=series.num_points,
        x_bounds=series.x_bounds,
        y_bounds=series.y_bounds,
    )


## } MODULE
