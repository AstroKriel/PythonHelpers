## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d_unstructured import gradient_operators
from jormi.ww_data import fit_series
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_plots, style_plots
from jormi.ww_types import box_positions

##
## === HELPER FUNCTIONS
##


def evaluate_curved_sarray(
    positions: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    """f(x, y, z) = sin(x) + cos(y) + 0.5 z^2; a smooth, genuinely curved field."""
    x_values, y_values, z_values = positions[:, 0], positions[:, 1], positions[:, 2]
    return numpy.sin(x_values) + numpy.cos(y_values) + 0.5 * z_values**2


def evaluate_exact_gradient(
    positions: numpy.ndarray[Any, numpy.dtype[Any]],
) -> numpy.ndarray[Any, numpy.dtype[Any]]:
    x_values, y_values, z_values = positions[:, 0], positions[:, 1], positions[:, 2]
    return numpy.stack(
        [numpy.cos(x_values), -numpy.sin(y_values), z_values],
        axis=1,
    )


def compute_typical_spacing(
    *,
    num_points: int,
    domain_volume: float,
) -> float:
    """Characteristic point spacing `\\tilde{\\Delta x} ~ (V / N)^(1/3)` for N points scattered in volume V."""
    return float((domain_volume / num_points)**(1.0 / 3.0))


##
## === NUMERICAL CONVERGENCE TEST
##


class TestGradientWLSConvergence:
    """
    WLS fits a local linear model, so it recovers a linear field exactly regardless of
    resolution (covered by the utest); a genuinely curved field is needed to see the
    approximation actually improve as point density increases. Points are scattered
    over a domain larger than the region where error is measured, so cells near the
    scored region have a full, unbiased neighbour stencil rather than a one-sided one.
    """

    def __init__(
        self,
    ):
        ## sample parameters
        self.seed: int = 42
        self.full_half_width: float = 4.0
        self.scored_half_width: float = 2.0
        self.k_neighbors: int = 20
        self.num_points_to_test: list[int] = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
        ## pass criterion: fitted convergence order must fall within these bounds; below the
        ## lower bound is no real convergence, above the upper suggests an overfit not real convergence
        self.convergence_order_bounds: tuple[float, float] = (0.3, 3.0)

    def run(
        self,
    ) -> None:
        fig, ax = manage_plots.create_figure(fig_scale=1.25)
        typical_spacings, rms_errors = self._measure_convergence()
        fitted_slope = self._plot_convergence(
            ax=ax,
            typical_spacings=typical_spacings,
            rms_errors=rms_errors,
        )
        file_path = Path(__file__).parent / "gradient_wls_convergence.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=file_path,
        )
        lower_bound, upper_bound = self.convergence_order_bounds
        converged = bool(lower_bound < fitted_slope.value < upper_bound)
        assert converged, (
            f"WLS gradient reconstruction did not converge as point density increased;"
            f" fitted order={fitted_slope.value:.2f} +/- {fitted_slope.sigma:.2f}."
        )
        manage_log.log_action(
            title="Gradient WLS convergence",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
            notes={"fitted order": f"{fitted_slope.value:.2f} +/- {fitted_slope.sigma:.2f}"},
        )

    def _measure_convergence(
        self,
    ) -> tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[Any]]]:
        rng = numpy.random.default_rng(seed=self.seed)
        domain_volume = (2.0 * self.full_half_width)**3
        typical_spacings: list[float] = []
        rms_errors: list[float] = []
        for num_points in self.num_points_to_test:
            positions = rng.uniform(
                -self.full_half_width,
                self.full_half_width,
                size=(num_points, 3),
            )
            values = evaluate_curved_sarray(positions)
            gradient = gradient_operators.compute_gradient_wls(
                positions,
                values,
                k_neighbors=self.k_neighbors,
            )
            ## score only points well inside the domain, so every scored point has a
            ## full, unbiased neighbour stencil rather than a one-sided one at the boundary
            scored_mask = numpy.all(
                numpy.abs(positions) < self.scored_half_width,
                axis=1,
            )
            exact_gradient = evaluate_exact_gradient(positions[scored_mask])
            rms_error = float(
                numpy.sqrt(
                    numpy.mean(
                        numpy.sum(
                            (gradient[scored_mask] - exact_gradient)**2,
                            axis=1,
                        ),
                    ),
                ),
            )
            typical_spacings.append(
                compute_typical_spacing(
                    num_points=num_points,
                    domain_volume=domain_volume,
                ),
            )
            rms_errors.append(rms_error)
        return numpy.array(typical_spacings), numpy.array(rms_errors)

    def _plot_convergence(
        self,
        *,
        ax: manage_plots.PlotAxis,
        typical_spacings: numpy.ndarray[Any, numpy.dtype[Any]],
        rms_errors: numpy.ndarray[Any, numpy.dtype[Any]],
    ) -> fit_series.FitStatistic:
        ## resolution increases rightward, matching `test_finite_difference_convergence.py`'s
        ## `inverse_dx_values` convention, rather than plotting spacing directly
        inverse_spacings = 1.0 / typical_spacings
        ## fit log(e) = slope * log(dx_tilde) + const; slope > 0 means error shrinks as dx_tilde shrinks
        log_series = GaussianSeries(
            x_values=numpy.log(typical_spacings),
            y_values=numpy.log(rms_errors),
        )
        fit = fit_series.fit_linear_model(log_series)
        fitted_slope = fit.slope
        fitted_errors = numpy.exp(fit.evaluate_fit(log_series.x_values))
        ax.plot(
            inverse_spacings,
            rms_errors,
            marker="o",
            ms=10,
            ls="",
            color="royalblue",
            label="measured",
        )
        ax.plot(
            inverse_spacings,
            fitted_errors,
            ls="--",
            lw=2,
            color="royalblue",
            label=rf"$e \sim O(\tilde{{\Delta x}}^{{{fitted_slope.value:.2f} \pm {fitted_slope.sigma:.2f}}})$",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/\tilde{\Delta x} \sim (N_{\rm points} / V)^{1/3}$")
        ax.set_ylabel(r"$e \equiv {\rm RMS}\,|\nabla f - \nabla f^*|$")
        ax.legend(loc=box_positions.MPLPositions.Anchor.Corner.TopRight)
        return fitted_slope


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    test = TestGradientWLSConvergence()
    test.run()

## } V-TEST
