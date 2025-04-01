## ###############################################################
## DEPENDENCIES
## ###############################################################
import sys
import numpy
from Loki.WWData import ComputeStats
from Loki.WWPlots import PlotUtils
from Loki.WWFields import FieldGradients


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def sample_domain(domain_bounds, num_points):
  return numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_points), endpoint=False) # to ensure periodicity

def evaluate_function_at_points(x_values):
  return numpy.sin(2 * x_values) + numpy.cos(x_values)

def evaluate_exact_function_derivative_at_points(x_values):
  return 2 * numpy.cos(2 * x_values) - numpy.sin(x_values)

def estimate_function_derivative(x_values, y_values, func_dydx):
  cell_width = (x_values[-1] - x_values[0]) / len(x_values) # assumes uniform sampling
  return func_dydx(y_values, cell_width, gradient_dir=0)

def calculate_powerlaw_amplitude(x_0, y_0, b):
  """Solve for the amplitude of a power law y = a * x^b given a coordinate (x_0, y_0) that the power-law passed through."""
  if x_0 == 0: return y_0
  return y_0 / numpy.power(x_0, b)


## ###############################################################
## NUMERICAL CONVERGENCE TEST
## ###############################################################
class TestFiniteDifferenceConvergence:
  def __init__(self):
    self.domain_bounds = [ 0, 2*numpy.pi ]
    self.num_samples_for_exact_soln  = 100
    self.num_samples_for_approx_soln = 15
    self.num_points_to_test = [ 10, 20, 50, 1e2, 2e2, 5e2 ]
    self.gradient_methods = [
      {"func": FieldGradients.gradient_2ocd, "expected_scaling": -2, "label": "2nd order", "color": "red"},
      {"func": FieldGradients.gradient_4ocd, "expected_scaling": -4, "label": "4th order", "color": "forestgreen"},
      {"func": FieldGradients.gradient_6ocd, "expected_scaling": -6, "label": "6th order", "color": "royalblue"}
    ]

  def run(self):
    fig, self.axs = PlotUtils.create_figure(num_rows=2, num_cols=2, fig_scale=1.35)
    self._plot_exact_soln()
    failed_methods = self._test_method_scaling()
    self._annotate_figure()
    PlotUtils.save_figure(fig, "finite_difference_convergence.png", bool_draft=False)
    assert len(failed_methods) == 0, f"Convergence test failed for the following method(s): {failed_methods}"
    print("Test passed successfully!")

  def _plot_exact_soln(self):
    x_values    = sample_domain(self.domain_bounds, self.num_samples_for_exact_soln)
    y_values    = evaluate_function_at_points(x_values)
    dydx_values = evaluate_exact_function_derivative_at_points(x_values)
    self.axs[0, 0].plot(x_values, y_values,    color="black", ls="-", lw=2)
    self.axs[1, 0].plot(x_values, dydx_values, color="black", ls="-", lw=2, label=r"${\rm d}y^* / {\rm d}x$")

  def _plot_approx_soln(self, grad_func, color, label):
    x_values    = sample_domain(self.domain_bounds, self.num_samples_for_approx_soln)
    y_values    = evaluate_function_at_points(x_values)
    dydx_values = estimate_function_derivative(x_values, y_values, grad_func)
    self.axs[1, 0].plot(x_values, dydx_values, marker="o", ms=10, ls="-", lw=2, color=color, label=label)

  def _test_method_scaling(self):
    failed_methods = []
    for method in self.gradient_methods:
      expected_scaling = method["expected_scaling"]
      grad_func        = method["func"]
      color            = method["color"]
      label            = method["label"]
      self._plot_approx_soln(grad_func, color, label)
      errors = []
      for num_points in self.num_points_to_test:
        x_values    = sample_domain(self.domain_bounds, num_points)
        y_values    = evaluate_function_at_points(x_values)
        dydx_exact  = evaluate_exact_function_derivative_at_points(x_values)
        dydx_approx = grad_func(y_values, numpy.diff(x_values).mean(), 0)
        error = ComputeStats.compute_p_norm(
          array_a             = dydx_exact,
          array_b             = dydx_approx,
          p_norm_order        = 2,
          normalise_by_length = True
        )
        errors.append(error)
      if not self._check_convergence(errors, expected_scaling, color, label):
        failed_methods.append(label)
    return failed_methods

  def _check_convergence(self, errors, expected_scaling, color, label):
    inverse_dx_values = numpy.array(self.num_points_to_test) / (self.domain_bounds[1] - self.domain_bounds[0])
    amplitude         = calculate_powerlaw_amplitude(inverse_dx_values[0], errors[0], expected_scaling)
    expected_errors   = amplitude * numpy.power(inverse_dx_values, expected_scaling)
    residuals         = (numpy.array(errors) - expected_errors) / expected_errors
    self.axs[0, 1].plot(
      inverse_dx_values,
      errors,
      marker="o", ms=10, ls="", color=color, label=label
    )
    self.axs[0, 1].plot(
      inverse_dx_values,
      expected_errors,
      ls="--", lw=2, color=color, label=f"$O(h^{{{expected_scaling}}})$",
      scalex=False, scaley=False
    )
    self.axs[1, 1].plot(
      inverse_dx_values[1:],
      numpy.abs(residuals[1:]),
      marker="o", ms=10, ls="-", lw=2, color=color
    )
    return numpy.all(numpy.diff(numpy.diff(numpy.abs(residuals[1:]))) < 0.0)

  def _annotate_figure(self):
    y_min, y_max = self.axs[1, 0].get_ylim()
    y_max_new = y_max + 0.2 * (y_max - y_min)
    self.axs[1, 0].set_ylim([y_min, y_max_new])
    self.axs[1, 0].text(
      0.5, 0.95,
      f"example with {self.num_samples_for_exact_soln} sampled points",
      ha        = "center",
      va        = "top",
      transform = self.axs[1, 0].transAxes
    )
    self.axs[0, 0].set_xticklabels([])
    self.axs[0, 0].set_ylabel(r"$y^*$")
    self.axs[1, 0].set_xlabel(r"$x$")
    self.axs[1, 0].set_ylabel(r"${\rm d}y/{\rm d}x$")
    self.axs[1, 0].legend(loc="lower right")
    self.axs[0, 1].set_xscale("log")
    self.axs[0, 1].set_yscale("log")
    self.axs[0, 1].set_xticklabels([])
    self.axs[0, 1].set_ylabel(r"$e_i \equiv (N)^{-1/2} \sum_{i=1}^N (y_i - y_i^*)^{1/2}$")
    self.axs[0, 1].legend(loc="lower left")
    self.axs[0, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
    self.axs[1, 1].set_xscale("log")
    self.axs[1, 1].set_yscale("log")
    self.axs[1, 1].set_xlabel(r"$1 / \Delta x = N / L$")
    self.axs[1, 1].set_ylabel(r"$|(e_i - e_i^*) / e_i^*|$")
    self.axs[1, 1].grid(True, which="both", linestyle="--", linewidth=0.5)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  test = TestFiniteDifferenceConvergence()
  test.run()
  sys.exit(0)


## END OF TEST