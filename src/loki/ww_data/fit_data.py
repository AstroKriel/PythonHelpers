## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from scipy.optimize import curve_fit


## ###############################################################
## FUNCTIONS
## ###############################################################
def fit_1d(
    data_x      : numpy.ndarray,
    data_y      : numpy.ndarray,
    index_start : int = 0,
    index_end   : int = None
  ) -> tuple[float, float]:
  """Fits a linear function to data using least squares optimization."""
  if index_end is None: index_end = len(data_x)
  x_fit = data_x[index_start:index_end]
  y_fit = data_y[index_start:index_end]
  func = lambda a0, a1, x : a0 + a1 * numpy.array(x)
  fitted_params, fit_covariance = curve_fit(func, x_fit, y_fit)
  gamma_val = fitted_params[1]
  gamma_std = numpy.sqrt(numpy.diag(fit_covariance))[1] if fit_covariance is not None else numpy.nan
  residual  = numpy.sum(numpy.square(y_fit - func(x_fit, *fitted_params)))
  return gamma_val, gamma_std, residual


## END OF MODULE