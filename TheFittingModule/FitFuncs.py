## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import warnings
import numpy as np
import statsmodels.api as sm

from scipy.optimize import curve_fit
from scipy import interpolate

## load user defined modules
from TheUsefulModule import WWLists, WWFuncs
from TheFittingModule import UserModels


## ###############################################################
## FUNCTIONS THAT INTERPOLATE AND FIT
## ###############################################################
def interpData(x, y, x_interp):
  return interpolate.CubicSpline(np.float32(x), np.float32(y))(x_interp)

def interpLogLogData(x, y, x_interp, interp_kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=interp_kind)
  y_interp = np.power(10.0, interpolator(np.log10(x_interp)))
  return y_interp

def fitExpFunc(
    data_x, data_y, index_start_fit, index_end_fit,
    ax                = None,
    num_interp_points = 10**2,
    color             = "black",
    linestyle         = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## uniformly sample interpolated data
  data_y_sampled = abs(interp_spline(data_fit_domain))
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_loge, fit_params_cov = curve_fit(
    f     = UserModels.ListOfModels.exp_loge,
    xdata = data_fit_domain,
    ydata = np.log(data_y_sampled)
  )
  gamma_std = np.sqrt(np.diag(fit_params_cov))[1]
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_loge[0]),
    fit_params_loge[1]
  ]
  gamma_val = fit_params_linear[1]
  if ax is not None:
    data_x_fit = np.linspace(-10, 500, 10**3)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      data_y_fit = UserModels.ListOfModels.exp_linear(data_x_fit, *fit_params_linear)
    ## find where exponential enters / exists fit range
    index_E_start = WWLists.getIndexClosestValue(data_x_fit, data_x[index_start_fit])
    index_E_end   = WWLists.getIndexClosestValue(data_x_fit, data_x[index_end_fit])
    str_label = r"$\gamma_{\rm exp} =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
    ax.plot(
      data_x_fit[index_E_start : index_E_end],
      data_y_fit[index_E_start : index_E_end],
      label=str_label, color=color, ls=linestyle, lw=2, zorder=5
    )
  return gamma_val, gamma_std

def fitLinearFunc(
    data_x, data_y,
    index_start_fit   = 0,
    index_end_fit     = None,
    ax                = None,
    num_interp_points = 10**2,
    color             = "black",
    linestyle         = "-"
  ):
  if index_end_fit is None: index_end_fit = len(data_x) - 1
  # ## make sure that the data is sorted
  # data_x, data_y = map(list, zip(*sorted(zip(data_x, data_y))))
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## uniformly sample interpolated data
  data_y_sampled = interp_spline(data_fit_domain)
  fit_params, fit_params_cov = curve_fit(
    f     = UserModels.ListOfModels.linear_offset,
    xdata = data_fit_domain,
    ydata = data_y_sampled
  )
  gamma_val = fit_params[1]
  gamma_std = np.sqrt(np.diag(fit_params_cov))[1]
  fit_error = np.sum((data_y_sampled - UserModels.ListOfModels.linear_offset(data_fit_domain, *fit_params))**2)
  if ax is not None:
    data_y_fit = UserModels.ListOfModels.linear_offset(data_fit_domain, *fit_params)
    index_E_start = WWLists.getIndexClosestValue(data_fit_domain, data_x[index_start_fit])
    index_E_end   = WWLists.getIndexClosestValue(data_fit_domain, data_x[index_end_fit])
    str_label = r"$\gamma_{\rm lin} =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
    ## find where exponential enters / exists fit range
    ax.plot(
      data_fit_domain[index_E_start : index_E_end],
      data_y_fit[index_E_start : index_E_end],
      label=str_label, color=color, ls=linestyle, lw=2, zorder=5
    )
  return gamma_val, fit_error

def fitConstFunc(
    data_x, data_y, index_start_fit, index_end_fit,
    ax                = None,
    num_interp_points = 10**2,
    str_label         = "",
    color             = "black",
    linestyle         = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    int(num_interp_points)
  )[1:-1]
  ## interpolate the non-uniform data
  interp_spline = interpolate.interp1d(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit],
    kind       = "cubic",
    fill_value = "extrapolate"
  )
  ## uniformly sample interpolated data
  data_y_sampled = interp_spline(data_fit_domain)
  fit_params, fit_params_cov = curve_fit(
    f     = UserModels.ListOfModels.constant,
    xdata = data_fit_domain,
    ydata = data_y_sampled
  )
  param_val = fit_params[0]
  param_std = np.sqrt(np.diag(fit_params_cov))[0]
  if ax is not None:
    data_x_fit = np.linspace(-10, 500, 10**3)
    data_y_fit = UserModels.ListOfModels.constant(data_x_fit, *fit_params)
    str_label += "{:.2f}".format(param_val) + r" $\pm$ " + "{:.2f}".format(param_std)
    index_E_start = WWLists.getIndexClosestValue(data_x_fit, data_x[index_start_fit])
    index_E_end   = WWLists.getIndexClosestValue(data_x_fit, data_x[index_end_fit])
    ax.plot(
      data_x_fit[index_E_start : index_E_end],
      data_y_fit[index_E_start : index_E_end],
      label=str_label, color=color, ls=linestyle, lw=2, zorder=5
    )
  ## return fitted quantities
  return param_val, param_std

@WWFuncs.timeFunc
def fitLineToMasked2DJPDF(bedges_cols, bedges_rows, jpdf, level):
  jpdf = np.array(jpdf)
  mg_rows, mg_cols = np.meshgrid(bedges_rows, bedges_cols, indexing="ij")
  mask = jpdf > level
  fit_obj = sm.OLS(mg_cols[1:,1:][mask], sm.add_constant(mg_rows[1:,1:][mask]))
  results = fit_obj.fit()
  intercept, slope = results.params
  return intercept, slope


## END OF LIBRARY