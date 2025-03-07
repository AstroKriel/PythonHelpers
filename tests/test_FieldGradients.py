## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt
from Loki.WWFields import FieldGradients
from Loki.WWStats import SimpleStats
from Loki.WWPlots import PlotUtils

plt.switch_backend("agg")


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getDomain(domain_bounds, num_points):
  return np.linspace(domain_bounds[0], domain_bounds[1], int(num_points), endpoint=False) # to ensure periodicity

def getData(array_x):
  # return np.sin(array_x)
  return np.sin(2*array_x) + np.cos(array_x)

def getExactDerivative(array_x):
  # return np.cos(array_x)
  return 2*np.cos(2*array_x) - np.sin(array_x)

def getApproxDerivative(data_x, data_y, func_dydx):
  cell_width = (data_x[-1] - data_x[0]) / len(data_x)
  return func_dydx(data_y, cell_width, gradient_dir=0)

def fitPowerlaw(x_0, y_0, b):
  """Solve for the amplitude of a power law y = a * x^b given a coordinate (x_0, y_0) that the power-law passed through."""
  if x_0 == 0: return y_0
  return y_0 / float(x_0)**(b)


## ###############################################################
## NUMERICAL CONVERGENCE TEST
## ###############################################################
def main():
  fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 3 * 5))
  dict_methods = [
    {"label": "2nd order", "order": -2, "func": FieldGradients.gradient_2ocd, "color": "red"},
    {"label": "4th order", "order": -4, "func": FieldGradients.gradient_4ocd, "color": "forestgreen"},
    {"label": "6th order", "order": -6, "func": FieldGradients.gradient_6ocd, "color": "royalblue"},
  ]
  num_samples      = 15
  domain_bounds    = [0, 2*np.pi]
  list_num_points  = [5, 10, 20, 50, 1e2, 2e2, 5e2, 1e3]
  array_x_approx   = getDomain(domain_bounds, num_samples)
  array_x_exact    = getDomain(domain_bounds, 100)
  array_y_approx   = getData(array_x_approx)
  array_y_exact    = getData(array_x_exact)
  array_dydx_exact = getExactDerivative(array_x_exact)
  axs[0].plot(array_x_exact, array_y_exact, "k-", lw=2, label="exact")
  axs[1].plot(array_x_exact, array_dydx_exact, "k-", lw=2, label="exact")
  for method in dict_methods:
    method_order = method["order"]
    method_func  = method["func"]
    method_color = method["color"]
    method_label = method["label"]
    dydx_approx  = getApproxDerivative(array_x_approx, array_y_approx, method_func)
    axs[1].plot(array_x_approx, dydx_approx, "o-", lw=1, ms=10, color=method_color, label=method_label)
    list_errors = []
    for num_points in list_num_points:
      x = getDomain(domain_bounds, num_points)
      y = getData(x)
      array_dydx_analytic = getExactDerivative(x)
      array_dydx_numeric = method_func(y, np.diff(x).mean(), 0)
      error = SimpleStats.computeNorm(array_dydx_numeric, array_dydx_analytic, p=2, bool_normalise=True)
      list_errors.append(error)
    array_inv_dx = np.array(list_num_points) / (domain_bounds[1] - domain_bounds[0])
    axs[2].loglog(array_inv_dx, list_errors, "o", ms=10, color=method_color, label=method_label)
    index_fit = np.argmax(array_inv_dx)
    amplitiude = fitPowerlaw(array_inv_dx[index_fit], list_errors[index_fit], method_order)
    array_fitted_errors = amplitiude * np.array(array_inv_dx, dtype=float)**(method_order)
    axs[2].loglog(
      array_inv_dx, array_fitted_errors,
      ls="--", lw=2, color=method_color, label=f"$O(h^{{{method_order}}})$"
    )
  axs[1].text(0.5, 0.95, f"example with {num_samples} sampled points", ha="center", va="top", transform=axs[1].transAxes)
  axs[0].set_xlabel(r"$x$")
  axs[1].set_xlabel(r"$x$")
  axs[2].set_xlabel(r"$1 / \Delta x = N / L$")
  axs[0].set_ylabel(r"$y(x)$")
  axs[1].set_ylabel(r"${\rm d}y/{\rm d}x$")
  axs[2].set_ylabel(r"$(N)^{-1/2} \sum_{i=1}^N (y_i - y_i^*)^{1/2}$")
  axs[1].legend(loc="lower right")
  axs[2].legend(loc="lower left")
  axs[2].grid(True, which="both", linestyle="--", linewidth=0.5)
  axs[2].set_xscale("log")
  axs[2].set_yscale("log")
  print("Saving figure...")
  PlotUtils.saveFigure(fig, "test_FieldGradients.png", bool_draft=False)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()

## END OF TEST
