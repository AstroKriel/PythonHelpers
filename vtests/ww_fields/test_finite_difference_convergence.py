## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any, Callable, TypedDict

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_arrays.farrays_3d import difference_sarrays
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_plots, style_plots

##
## === TYPE ALIASES
##


class _GradMethod(TypedDict):
    worker_fn: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]]
    expected_scaling: int
    label: str
    color: str


##
## === HELPER FUNCTIONS
##


def sample_domain(
    domain_bounds: list[float],
    num_points: float,
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return numpy.linspace(
        domain_bounds[0],
        domain_bounds[1],
        int(num_points),
        endpoint=False,
    )  # to ensure periodicity


def evaluate_function(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return numpy.sin(2 * x_values) + numpy.cos(x_values)


def evaluate_exact_function_derivative(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return 2 * numpy.cos(2 * x_values) - numpy.sin(x_values)


def estimate_function_derivative(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
    y_values: numpy.ndarray[Any, numpy.dtype[Any]],
    func_dydx: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    cell_width = (x_values[-1] - x_values[0]) / len(x_values)  # assumes uniform samples
    return func_dydx(
        sarray_3d=y_values[:, None, None],
        cell_width=cell_width,
        grad_axis=0,
    )[:, 0, 0]


def calculate_powerlaw_amplitude(
    x_0: float,
    y_0: float,
    b: float,
) -> float:
    """Solve for the amplitude of a power law y = a * x^b given a coordinate (x_0, y_0) that the power-law passed through."""
    if x_0 == 0:
        return y_0
    return float(y_0 / numpy.power(x_0, b))


##
## === NUMERICAL CONVERGENCE TEST
##


class TestFiniteDifferenceConvergence:

    def __init__(
        self,
    ):
        ## sample parameters
        self.domain_bounds: list[float] = [0, 2 * numpy.pi]
        self.num_samples_for_exact_soln: int = 100
        self.num_samples_for_approx_soln: int = 15
        self.num_points_to_test: list[float] = [10, 20, 50, 1e2, 2e2, 5e2]
        ## scenario config: the finite-difference methods to test and their expected scalings
        self.grad_methods: list[_GradMethod] = [
            {
                "worker_fn": difference_sarrays.second_order_centered_difference,
                "expected_scaling": -2,
                "label": "2nd order",
                "color": "red",
            },
            {
                "worker_fn": difference_sarrays.fourth_order_centered_difference,
                "expected_scaling": -4,
                "label": "4th order",
                "color": "forestgreen",
            },
            {
                "worker_fn": difference_sarrays.sixth_order_centered_difference,
                "expected_scaling": -6,
                "label": "6th order",
                "color": "royalblue",
            },
        ]

    def run(
        self,
    ) -> None:
        fig, axs_grid = manage_plots.create_figure(
            num_rows=2,
            num_cols=2,
            fig_scale=2.0,
            x_spacing=0.35,
        )
        self._plot_exact_soln(axs_grid)
        failed_methods = self._test_method_scaling(axs_grid)
        self._annotate_figure(axs_grid)
        file_name = "finite_difference_convergence.png"
        file_path = Path(__file__).parent / file_name
        manage_plots.save_figure(
            fig=fig,
            fig_path=file_path,
        )
        assert len(
            failed_methods,
        ) == 0, f"Convergence test failed for the following method(s): {ww_lists.as_string(elems=failed_methods)}"
        manage_log.log_action(
            title="Finite difference convergence",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
        )

    def _plot_exact_soln(
        self,
        axs_grid: manage_plots.PlotAxesGrid,
    ) -> None:
        x_values = sample_domain(self.domain_bounds, self.num_samples_for_exact_soln)
        y_values = evaluate_function(x_values)
        dydx_values = evaluate_exact_function_derivative(x_values)
        axs_grid[0, 0].plot(
            x_values,
            y_values,
            color="black",
            ls="-",
            lw=2,
        )
        axs_grid[1, 0].plot(
            x_values,
            dydx_values,
            color="black",
            ls="-",
            lw=2,
            label=r"${\rm d}y^* / {\rm d}x$",
        )

    def _plot_approx_soln(
        self,
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        nabla: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]],
        color: str,
        label: str,
    ) -> None:
        x_values = sample_domain(self.domain_bounds, self.num_samples_for_approx_soln)
        y_values = evaluate_function(x_values)
        dydx_values = estimate_function_derivative(x_values, y_values, nabla)
        axs_grid[1, 0].plot(
            x_values,
            dydx_values,
            marker="o",
            ms=10,
            ls="-",
            lw=2,
            color=color,
            label=label,
        )

    def _test_method_scaling(
        self,
        axs_grid: manage_plots.PlotAxesGrid,
    ) -> list[str]:
        failed_methods: list[str] = []
        for grad_method in self.grad_methods:
            expected_scaling = grad_method["expected_scaling"]
            nabla = grad_method["worker_fn"]
            color = grad_method["color"]
            label = grad_method["label"]
            self._plot_approx_soln(
                axs_grid=axs_grid,
                nabla=nabla,
                color=color,
                label=label,
            )
            rms_errors: list[float] = []
            for num_points in self.num_points_to_test:
                x_values = sample_domain(self.domain_bounds, num_points)
                y_values = evaluate_function(x_values)
                dydx_exact = evaluate_exact_function_derivative(x_values)
                cell_width = x_values[1] - x_values[0]  # assumes uniform samples
                dydx_approx = nabla(
                    sarray_3d=y_values[:, None, None],
                    cell_width=cell_width,
                    grad_axis=0,
                )[:, 0, 0]
                rms_error = float(
                    numpy.sqrt(
                        numpy.mean(
                            numpy.square(
                                dydx_exact - dydx_approx,
                            ),
                        ),
                    ),
                )
                rms_errors.append(rms_error)
            has_converged = self._check_convergence(
                axs_grid=axs_grid,
                rms_errors=rms_errors,
                expected_scaling=expected_scaling,
                color=color,
                label=label,
            )
            if not has_converged:
                failed_methods.append(label)
            manage_log.log_outcome(
                text=f"{label} (expected scaling O(h^{expected_scaling}))",
                outcome=(
                    manage_log.ActionOutcome.SUCCESS
                    if has_converged
                    else manage_log.ActionOutcome.FAILURE
                ),
            )
        return failed_methods

    def _check_convergence(
        self,
        *,
        axs_grid: manage_plots.PlotAxesGrid,
        rms_errors: list[float],
        expected_scaling: int,
        color: str,
        label: str,
    ) -> bool:
        inverse_dx_values = numpy.array(
            self.num_points_to_test,
        ) / (self.domain_bounds[1] - self.domain_bounds[0])
        amplitude = fit_series.get_powerlaw_amplitude(
            exponent=float(expected_scaling),
            x_ref=float(inverse_dx_values[0]),
            y_ref=rms_errors[0],
        )
        expected_errors = amplitude * numpy.power(inverse_dx_values, expected_scaling)
        residuals = (numpy.array(rms_errors) - expected_errors) / expected_errors
        axs_grid[0, 1].plot(
            inverse_dx_values,
            rms_errors,
            marker="o",
            ms=10,
            ls="",
            color=color,
            label=label,
        )
        axs_grid[0, 1].plot(
            inverse_dx_values,
            expected_errors,
            ls="--",
            lw=2,
            color=color,
            label=rf"$e_i^* \sim O(h^{{{expected_scaling}}})$",
            scalex=False,
            scaley=False,
        )
        axs_grid[1, 1].plot(
            inverse_dx_values[1:],
            numpy.abs(residuals[1:]),
            marker="o",
            ms=10,
            ls="-",
            lw=2,
            color=color,
        )
        return bool(
            numpy.all(
                numpy.diff(
                    numpy.diff(
                        numpy.abs(
                            residuals[1:],
                        ),
                    ),
                ) < 0.0,
            ),
        )

    def _annotate_figure(
        self,
        axs_grid: manage_plots.PlotAxesGrid,
    ) -> None:
        y_min, y_max = axs_grid[1, 0].get_ylim()
        y_max_new = y_max + 0.2 * (y_max - y_min)
        axs_grid[1, 0].set_ylim([y_min, y_max_new])
        axs_grid[1, 0].text(
            0.5,
            0.95,
            f"example with {self.num_samples_for_exact_soln} sampled points",
            ha="center",
            va="top",
            transform=axs_grid[1, 0].transAxes,
        )
        axs_grid[0, 0].set_xticklabels([])
        axs_grid[0, 0].set_ylabel(r"$y^*$")
        axs_grid[1, 0].set_xlabel(r"$x$")
        axs_grid[1, 0].set_ylabel(r"${\rm d}y/{\rm d}x$")
        axs_grid[1, 0].legend(loc="lower right")
        axs_grid[0, 1].set_xscale("log")
        axs_grid[0, 1].set_yscale("log")
        axs_grid[0, 1].set_xticklabels([])
        axs_grid[0, 1].set_ylabel(r"$e_i \equiv (N)^{-1/2} \sum_{i=1}^N (y_i - y_i^*)^{1/2}$")
        axs_grid[0, 1].legend(loc="lower left")
        axs_grid[0, 1].grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
        )
        axs_grid[1, 1].set_xscale("log")
        axs_grid[1, 1].set_yscale("log")
        axs_grid[1, 1].set_xlabel(r"$1 / \Delta x = N / L$")
        axs_grid[1, 1].set_ylabel(r"$|(e_i - e_i^*) / e_i^*|$")
        axs_grid[1, 1].grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    test = TestFiniteDifferenceConvergence()
    test.run()

## } V-TEST
