## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import SmoothData
from Loki.WWPlots import PlotUtils
from Loki.WWFields import FieldGradients


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getDomain(domain_bounds, num_points):
  return numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_points), endpoint=False) # to ensure periodicity

def getData(array_x):
  return numpy.sin(2*array_x) + numpy.cos(array_x)

def getExactDerivative(array_x):
  return 2*numpy.cos(2*array_x) - numpy.sin(array_x)

def getApproxDerivative(data_x, data_y, func_dydx):
  cell_width = (data_x[-1] - data_x[0]) / len(data_x)
  return func_dydx(data_y, cell_width, gradient_dir=0)

def fitPowerlawAmplitude(x_0, y_0, b):
  """Solve for the amplitude of a power law y = a * x^b given a coordinate (x_0, y_0) that the power-law passed through."""
  if x_0 == 0: return y_0
  return y_0 / numpy.power(x_0, b)


## ###############################################################
## NUMERICAL CONVERGENCE TEST
## ###############################################################
def main():
  fig, axs = PlotUtils.create_figure(num_rows=2, num_cols=2, fig_scale=1.35)
  dict_methods = [
    {"label": "2nd order", "order": -2, "func": FieldGradients.gradient_2ocd, "color": "red"},
    {"label": "4th order", "order": -4, "func": FieldGradients.gradient_4ocd, "color": "forestgreen"},
    {"label": "6th order", "order": -6, "func": FieldGradients.gradient_6ocd, "color": "royalblue"},
  ]
  num_samples      = 15
  domain_bounds    = [ 0, 2*numpy.pi ]
  list_num_points  = [ 10, 20, 50, 1e2, 2e2, 5e2 ]
  array_x_approx   = getDomain(domain_bounds, num_samples)
  array_x_exact    = getDomain(domain_bounds, 100)
  array_y_approx   = getData(array_x_approx)
  array_y_exact    = getData(array_x_exact)
  array_dydx_exact = getExactDerivative(array_x_exact)
  axs[0,0].plot(array_x_exact, array_y_exact, "k-", lw=2)
  axs[1,0].plot(array_x_exact, array_dydx_exact, "k-", lw=2, label=r"${\rm d}y^* / {\rm d}x$")
  list_failed_methods = []
  for method in dict_methods:
    method_order = method["order"]
    method_func  = method["func"]
    method_color = method["color"]
    method_label = method["label"]
    dydx_approx  = getApproxDerivative(array_x_approx, array_y_approx, method_func)
    axs[1,0].plot(array_x_approx, dydx_approx, "o-", lw=1, ms=10, color=method_color, label=method_label)
    list_errors = []
    for num_points in list_num_points:
      x = getDomain(domain_bounds, num_points)
      y = getData(x)
      array_dydx_analytic = getExactDerivative(x)
      array_dydx_numeric = method_func(y, numpy.diff(x).mean(), 0)
      error = SmoothData.compute_p_norm(array_dydx_numeric, array_dydx_analytic, p=2, bool_normalise=True)
      list_errors.append(error)
    array_inv_dx = numpy.array(list_num_points) / (domain_bounds[1] - domain_bounds[0])
    axs[0,1].plot(array_inv_dx, list_errors, "o", ms=10, color=method_color, label=method_label)
    amplitiude = fitPowerlawAmplitude(array_inv_dx[0], list_errors[0], method_order)
    array_expected_errors = amplitiude * numpy.array(array_inv_dx, dtype=float)**(method_order)
    array_residuals = (numpy.array(list_errors) - array_expected_errors) / array_expected_errors
    axs[0,1].plot(array_inv_dx, array_expected_errors, ls="--", lw=2, color=method_color, label=f"$O(h^{{{method_order}}})$")
    axs[1,1].plot(array_inv_dx[1:], numpy.abs(array_residuals[1:]), marker="o", ms=10, color=method_color)
    ## check that the difference between the errors are getting smaller
    if not numpy.all(numpy.diff(numpy.diff(numpy.abs(array_residuals[1:]))) < 0.0):
      list_failed_methods.append(method_label)
  dydx_min = numpy.min(array_dydx_exact)
  dydx_max = numpy.max(array_dydx_exact)
  axs[1,0].text(0.5, 0.95, f"example with {num_samples} sampled points", ha="center", va="top", transform=axs[1,0].transAxes)
  axs[1,0].set_ylim([ 
    1.2 * dydx_min if (numpy.sign(dydx_min) < 0.0) else 0.8 * dydx_min,
    0.8 * dydx_max if (numpy.sign(dydx_max) < 0.0) else 1.65 * dydx_max
  ])
  axs[0,0].set_xticklabels([])
  axs[0,0].set_ylabel(r"$y^*$")
  axs[1,0].set_xlabel(r"$x$")
  axs[1,0].set_ylabel(r"${\rm d}y/{\rm d}x$")
  axs[1,0].legend(loc="lower right")
  inv_dx_bounds = [ 0.9*numpy.min(array_inv_dx), 1.1*numpy.max(array_inv_dx) ]
  axs[0,1].set_xlim(inv_dx_bounds)
  axs[0,1].set_xscale("log")
  axs[0,1].set_yscale("log")
  axs[0,1].set_xticklabels([])
  axs[0,1].set_ylabel(r"$e_i \equiv (N)^{-1/2} \sum_{i=1}^N (y_i - y_i^*)^{1/2}$")
  axs[0,1].legend(loc="lower left")
  axs[0,1].grid(True, which="both", linestyle="--", linewidth=0.5)
  axs[1,1].set_xlim(inv_dx_bounds)
  axs[1,1].set_xscale("log")
  axs[1,1].set_yscale("log")
  axs[1,1].set_xlabel(r"$1 / \Delta x = N / L$")
  axs[1,1].set_ylabel(r"$|(e_i - e_i^*) / e_i^*|$")
  axs[1,1].grid(True, which="both", linestyle="--", linewidth=0.5)
  print("Saving figure...")
  PlotUtils.save_figure(fig, "finite_difference_convergence.png", bool_draft=False)
  assert len(list_failed_methods) == 0, f"Convergence test failed for the following method(s): {list_failed_methods}"
  print("Test passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST