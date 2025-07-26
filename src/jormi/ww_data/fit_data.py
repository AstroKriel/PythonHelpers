## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from scipy.optimize import curve_fit


## ###############################################################
## FUNCTIONS
## ###############################################################

def linear_function(intercept, slope, x_values):
  return intercept + slope * numpy.array(x_values)

def get_linear_intercept(slope, x_ref, y_ref):
  return y_ref - slope * x_ref

def get_powerlaw_coefficient(exponent, x_ref, y_ref):
  return y_ref / x_ref**exponent

def get_line_angle_in_box(slope, domain_bounds, domain_aspect_ratio=1.0):
  x_min, x_max, y_min, y_max = domain_bounds
  data_aspect_ratio = (x_max - x_min) / (y_max - y_min)
  scale_x = 1
  scale_y = data_aspect_ratio / domain_aspect_ratio
  delta_x = 1 * scale_x
  delta_y = slope * scale_y
  angle_rad = numpy.arctan2(delta_y, delta_x)
  angle_deg = angle_rad * 180 / numpy.pi
  return angle_deg

def fit_1d_linear_model(
    x_values    : list | numpy.ndarray,
    y_values    : list | numpy.ndarray,
    index_start : int = 0,
    index_end   : int | None = None
  ) -> dict:
  """Fits a linear function to data using least squares optimization."""
  if index_end is None: index_end = len(x_values)
  if len(x_values) != len(y_values): raise ValueError("`x_values` and `y_values` must have the same length.")
  ## note: truncates values locally: if input arguments are lists, then they will not be mutated
  x_values = x_values[index_start:index_end]
  y_values = y_values[index_start:index_end]
  fitted_params, fit_covariance = curve_fit(linear_function, x_values, y_values)
  intercept, slope = fitted_params
  if fit_covariance is not None:
    intercept_std, slope_std = numpy.sqrt(numpy.diag(fit_covariance))
  else: intercept_std, slope_std = (None, None)
  residual = numpy.sum(numpy.square(y_values - linear_function(x_values, *fitted_params)))
  return {
    "intercept": {
      "best": intercept,
      "std": intercept_std
    },
    "slope": {
      "best": slope,
      "std": slope_std
    },
    "residual": residual
  }


## END OF MODULE