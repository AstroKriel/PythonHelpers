## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_arrays import compute_array_stats
from jormi.ww_arrays.farrays_3d import difference_sarrays
from jormi.ww_data import fit_series
from jormi.ww_io import manage_log
from jormi.ww_plots import annotate_panel, manage_plots, style_plots
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##


@dataclass(frozen=True)
class FiniteDifferenceMethod:
    dydx_fn: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]]
    expected_scaling: int
    label: str
    color: str


##
## === HELPER FUNCTIONS
##


def sample_domain(
    *,
    domain_bounds: list[float],
    num_points: float,
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return numpy.linspace(
        domain_bounds[0],
        domain_bounds[1],
        int(num_points),
        endpoint=False,
    )  # to ensure periodicity


def evaluate_model(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return numpy.sin(2 * x_values) + numpy.cos(x_values)


def evaluate_exact_dydx(
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    return 2 * numpy.cos(2 * x_values) - numpy.sin(x_values)


def evaluate_approx_dydx(
    *,
    x_values: numpy.ndarray[Any, numpy.dtype[Any]],
    y_values: numpy.ndarray[Any, numpy.dtype[Any]],
    dydx_fn: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    cell_width = x_values[1] - x_values[0]  # assumes uniform samples
    return dydx_fn(
        sarray_3d=y_values[:, None, None],
        cell_width=cell_width,
        grad_axis=0,
    )[:, 0, 0]


def residual_is_plateauing(
    residuals: numpy.ndarray[Any, numpy.dtype[Any]],
) -> bool:
    residual_magnitudes = numpy.abs(residuals[1:])
    return bool(numpy.all(numpy.diff(numpy.diff(residual_magnitudes)) < 0.0))


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
        self.grad_methods: list[FiniteDifferenceMethod] = [
            FiniteDifferenceMethod(
                dydx_fn=difference_sarrays.second_order_centered_difference,
                expected_scaling=-2,
                label="2nd order",
                color="red",
            ),
            FiniteDifferenceMethod(
                dydx_fn=difference_sarrays.fourth_order_centered_difference,
                expected_scaling=-4,
                label="4th order",
                color="forestgreen",
            ),
            FiniteDifferenceMethod(
                dydx_fn=difference_sarrays.sixth_order_centered_difference,
                expected_scaling=-6,
                label="6th order",
                color="royalblue",
            ),
        ]

    def run(
        self,
    ) -> None:
        fig, panels_grid = manage_plots.create_figure(
            num_rows=2,
            num_cols=2,
            figure_scale=2.0,
            x_spacing=0.35,
        )
        self._plot_exact_soln(panels_grid)
        failed_methods = self._test_method_scaling(panels_grid)
        self._annotate_figure(panels_grid)
        file_name = "finite_difference_convergence.png"
        file_path = Path(__file__).parent / file_name
        manage_plots.save_figure(
            fig=fig,
            figure_path=file_path,
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
        panels_grid: manage_plots.PlotPanelGrid,
    ) -> None:
        x_values = sample_domain(
            domain_bounds=self.domain_bounds,
            num_points=self.num_samples_for_exact_soln,
        )
        y_values = evaluate_model(x_values)
        dydx_values = evaluate_exact_dydx(x_values)
        panels_grid[0, 0].plot(
            x_values,
            y_values,
            color="black",
            ls="-",
            lw=2,
        )
        panels_grid[1, 0].plot(
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
        panels_grid: manage_plots.PlotPanelGrid,
        dydx_fn: Callable[..., numpy.ndarray[Any, numpy.dtype[Any]]],
        color: str,
        label: str,
    ) -> None:
        x_values = sample_domain(
            domain_bounds=self.domain_bounds,
            num_points=self.num_samples_for_approx_soln,
        )
        y_values = evaluate_model(x_values)
        dydx_values = evaluate_approx_dydx(
            x_values=x_values,
            y_values=y_values,
            dydx_fn=dydx_fn,
        )
        panels_grid[1, 0].plot(
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
        panels_grid: manage_plots.PlotPanelGrid,
    ) -> list[str]:
        failed_methods: list[str] = []
        for grad_method in self.grad_methods:
            expected_scaling = grad_method.expected_scaling
            dydx_fn = grad_method.dydx_fn
            color = grad_method.color
            label = grad_method.label
            self._plot_approx_soln(
                panels_grid=panels_grid,
                dydx_fn=dydx_fn,
                color=color,
                label=label,
            )
            rms_errors: list[float] = []
            for num_points in self.num_points_to_test:
                x_values = sample_domain(
                    domain_bounds=self.domain_bounds,
                    num_points=num_points,
                )
                y_values = evaluate_model(x_values)
                dydx_exact = evaluate_exact_dydx(x_values)
                dydx_approx = evaluate_approx_dydx(
                    x_values=x_values,
                    y_values=y_values,
                    dydx_fn=dydx_fn,
                )
                rms_error = compute_array_stats.compute_rms(dydx_exact - dydx_approx)
                rms_errors.append(rms_error)
            has_converged = self._check_convergence(
                panels_grid=panels_grid,
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
        panels_grid: manage_plots.PlotPanelGrid,
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
        panels_grid[0, 1].plot(
            inverse_dx_values,
            rms_errors,
            marker="o",
            ms=10,
            ls="",
            color=color,
            label=label,
        )
        panels_grid[0, 1].plot(
            inverse_dx_values,
            expected_errors,
            ls="--",
            lw=2,
            color=color,
            label=rf"$e_i^* \sim O(h^{{{expected_scaling}}})$",
            scalex=False,
            scaley=False,
        )
        panels_grid[1, 1].plot(
            inverse_dx_values[1:],
            numpy.abs(residuals[1:]),
            marker="o",
            ms=10,
            ls="-",
            lw=2,
            color=color,
        )
        return residual_is_plateauing(residuals)

    def _annotate_figure(
        self,
        panels_grid: manage_plots.PlotPanelGrid,
    ) -> None:
        y_min, y_max = panels_grid[1, 0].get_ylim()
        y_max_new = y_max + 0.2 * (y_max - y_min)
        panels_grid[1, 0].set_ylim([y_min, y_max_new])
        annotate_panel.add_text(
            panel=panels_grid[1, 0],
            x_pos=0.5,
            y_pos=0.95,
            label=f"example with {self.num_samples_for_exact_soln} sampled points",
            x_alignment=box_positions.Positions.Center.Center,
            y_alignment=box_positions.Positions.Side.Top,
        )
        panels_grid[0, 0].set_xticklabels([])
        panels_grid[0, 0].set_ylabel(r"$y^*$")
        panels_grid[1, 0].set_xlabel(r"$x$")
        panels_grid[1, 0].set_ylabel(r"${\rm d}y/{\rm d}x$")
        panels_grid[1, 0].legend(loc="lower right")
        panels_grid[0, 1].set_xscale("log")
        panels_grid[0, 1].set_yscale("log")
        panels_grid[0, 1].set_xticklabels([])
        panels_grid[0, 1].set_ylabel(r"$e_i \equiv (N)^{-1/2} \sum_{i=1}^N (y_i - y_i^*)^{1/2}$")
        panels_grid[0, 1].legend(loc="lower left")
        panels_grid[0, 1].grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
        )
        panels_grid[1, 1].set_xscale("log")
        panels_grid[1, 1].set_yscale("log")
        panels_grid[1, 1].set_xlabel(r"$1 / \Delta x = N / L$")
        panels_grid[1, 1].set_ylabel(r"$|(e_i - e_i^*) / e_i^*|$")
        panels_grid[1, 1].grid(
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
