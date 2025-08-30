## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from scipy.optimize import curve_fit


## ###############################################################
## FUNCTIONS
## ###############################################################

def get_linear_intercept(
  slope : float,
  x_ref : float,
  y_ref : float,
):
  return y_ref - slope * x_ref

def get_powerlaw_coefficient(
  exponent : float,
  x_ref    : float,
  y_ref    : float,
):
  return y_ref / x_ref**exponent

def get_line_angle(
  slope               : float,
  domain_bounds       : tuple[float, float, float, float],
  domain_aspect_ratio : float = 1.0,
):
  x_min, x_max, y_min, y_max = domain_bounds
  data_aspect_ratio = (x_max - x_min) / (y_max - y_min)
  scale_x = 1
  scale_y = data_aspect_ratio / domain_aspect_ratio
  delta_x = 1 * scale_x
  delta_y = slope * scale_y
  angle_rad = numpy.arctan2(delta_y, delta_x)
  angle_deg = angle_rad * 180 / numpy.pi
  return angle_deg

def linear_function(
  x_values  : list | numpy.ndarray,
  intercept : float,
  slope     : float,
):
  return intercept + slope * numpy.array(x_values)

def fit_1d_linear_model(
  x_values    : list | numpy.ndarray,
  y_values    : list | numpy.ndarray,
  index_start : int = 0,
  index_end   : int | None = None,
) -> dict:
  """Fits a linear function to data using least squares optimization."""
  x_values = numpy.asarray(x_values, dtype=float)
  y_values = numpy.asarray(y_values, dtype=float)
  if index_end is None:
    index_end = len(x_values)
  if len(x_values) != len(y_values):
    raise ValueError("`x_values` and `y_values` must have the same length.")
  ## note: truncates values locally. even if the input args are lists, they will not be mutated globally
  x_values = x_values[index_start:index_end]
  y_values = y_values[index_start:index_end]
  fitted_params, fit_covariance = curve_fit(linear_function, x_values, y_values)
  intercept, slope = fitted_params
  if fit_covariance is not None:
    intercept_std, slope_std = numpy.sqrt(numpy.diag(fit_covariance))
  else: intercept_std, slope_std = (None, None)
  residual = y_values - linear_function(x_values, *fitted_params)
  ssr = numpy.sum(numpy.square(residual))
  return {
    "intercept": {
      "best": intercept,
      "std": intercept_std
    },
    "slope": {
      "best": slope,
      "std": slope_std
    },
    "residual": residual,
    "ssr": ssr,
  }

def fit_line_with_fixed_slope(
  x_values : list | numpy.ndarray,
  y_values : list | numpy.ndarray,
  slope    : float,
  y_sigmas : list | numpy.ndarray | None = None,
) -> dict:
  x_values = numpy.asarray(x_values, dtype=float)
  y_values = numpy.asarray(y_values, dtype=float)
  if len(x_values) != len(y_values):
    raise ValueError("`x_values` and `y_values` must have the same length.")
  num_values = len(x_values)
  if num_values < 2:
    raise ValueError("Need at least 2 points to estimate std.")
  if y_sigmas is None:
    y_sigmas = numpy.ones_like(y_values)
  elif len(y_sigmas) != len(y_values):
    raise ValueError("`y_sigmas` and `y_values` must have the same length.")
  weights = 1.0 / numpy.square(y_sigmas)
  intercept_best = (
      numpy.sum(weights * y_values) - slope * numpy.sum(weights * x_values)
    ) / numpy.sum(weights)
  residual_values = y_values - (intercept_best + slope * x_values)
  ssr = numpy.sum(numpy.square(residual_values))
  sigma_squared = ssr / (num_values - 1)
  intercept_std = numpy.sqrt(sigma_squared / num_values)
  return {
    "intercept": {
      "best": intercept_best,
      "std": intercept_std
    },
    "slope": {"best": slope, "std": None},
    "residual": residual_values,
    "ssr": ssr,
  }


## END OF MODULE