## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_data import fit_series
from jormi.ww_data import series_types
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_figure, style_figure

##
## === TYPE ALIASES
##


@dataclass(frozen=True)
class FitScenario:
    label: str
    fit: fit_series.LinearFitSummary


##
## === HELPER FUNCTIONS
##


def plot_fit(
    *,
    panel: manage_figure.Panel,
    gaussian_series: series_types.GaussianSeries,
    fit: fit_series.LinearFitSummary,
    fit_label: str,
    fit_index: int,
    num_fits: int,
) -> None:
    is_top_panel = fit_index == 0
    is_bottom_panel = fit_index == num_fits - 1
    x_fit_values = numpy.linspace(gaussian_series.x_bounds[0], gaussian_series.x_bounds[1], 200)
    panel.errorbar(
        gaussian_series.x_values,
        gaussian_series.y_values,
        yerr=gaussian_series.y_sigmas,
        fmt="o",
        color="black",
        label="data" if is_top_panel else None,
    )
    panel.plot(
        x_fit_values,
        fit.evaluate_fit(x_fit_values),
        color="red",
        label=fit_label,
    )
    panel.set_ylabel("y")
    panel.legend(
        fontsize=20,
        loc="upper left",
    )
    if is_bottom_panel:
        panel.set_xlabel("x")
    else:
        panel.tick_params(labelbottom=False)


##
## === FIT ACCURACY TEST
##


class TestLinearFit:
    """
    Fit a noisy linear dataset and check that each fit method recovers the true
    slope and intercept to within `sigma_tol` of its reported uncertainty.
    """

    def __init__(
        self,
    ):
        ## sample parameters
        self.seed: int = 42
        self.num_points: int = 20
        self.noise_sigma: float = 1.5
        ## pass criterion: recovered params must lie within sigma_tol of truth
        self.sigma_tol: float = 3.0
        ## ground-truth model: y = true_slope * x + true_intercept
        self.true_slope: float = 2.5
        self.true_intercept: float = 1.0

    def run(
        self,
    ) -> None:
        gaussian_series = self._generate_gaussian_series()
        fits_to_test = self._compute_fits(gaussian_series)
        num_fits = len(fits_to_test)
        figure, panel_grid = manage_figure.create_figure(
            num_panel_rows=num_fits,
            num_panel_columns=1,
            share_x_axis=True,
        )
        failed_fits: list[str] = []
        for fit_index, fit_scenario in enumerate(fits_to_test):
            panel = panel_grid[fit_index, 0]
            plot_fit(
                panel=panel,
                gaussian_series=gaussian_series,
                fit=fit_scenario.fit,
                fit_label=fit_scenario.label,
                fit_index=fit_index,
                num_fits=num_fits,
            )
            failed_checks = self._find_failed_checks(fit_scenario.fit)
            if failed_checks:
                for check_msg in failed_checks:
                    manage_log.log_outcome(
                        text=f"{fit_scenario.label}: {check_msg}",
                        outcome=manage_log.ActionOutcome.FAILURE,
                    )
                failed_fits.append(fit_scenario.label)
            else:
                manage_log.log_outcome(
                    text=(
                        f"{fit_scenario.label} (slope={fit_scenario.fit.slope.value:.4f}, "
                        f"intercept={fit_scenario.fit.intercept.value:.4f})"
                    ),
                    outcome=manage_log.ActionOutcome.SUCCESS,
                )
        ## always save even on failure
        figure_path = Path(__file__).parent / "linear_fit.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
        )
        assert not failed_fits, (
            f"Test failed for the following fit methods: {ww_lists.as_string(elems=failed_fits)}"
        )
        manage_log.log_action(
            title="Linear fit",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
        )

    def _generate_gaussian_series(
        self,
    ) -> series_types.GaussianSeries:
        rng = numpy.random.default_rng(seed=self.seed)
        x_values = numpy.linspace(0.0, 10.0, self.num_points)
        y_sigmas = self.noise_sigma * numpy.ones_like(x_values)
        y_values = self.true_slope * x_values + self.true_intercept + rng.normal(scale=y_sigmas)
        return series_types.GaussianSeries(
            x_values=x_values,
            y_values=y_values,
            y_sigmas=y_sigmas,
        )

    def _compute_fits(
        self,
        gaussian_series: series_types.GaussianSeries,
    ) -> list[FitScenario]:
        return [
            FitScenario(
                label="linear model",
                fit=fit_series.fit_linear_model(gaussian_series),
            ),
            FitScenario(
                label="fixed slope",
                fit=fit_series.fit_line_with_fixed_slope(
                    gaussian_series=gaussian_series,
                    fixed_slope=self.true_slope,
                ),
            ),
        ]

    def _find_failed_checks(
        self,
        fit: fit_series.LinearFitSummary,
    ) -> list[str]:
        fitted_slope = fit.slope
        fitted_intercept = fit.intercept
        slope_error = abs(fitted_slope.value - self.true_slope)
        intercept_error = abs(fitted_intercept.value - self.true_intercept)
        ## skip a param whose sigma is unavailable
        failed_checks: list[str] = []
        if (fitted_slope.sigma is not None) and (slope_error > self.sigma_tol * fitted_slope.sigma):
            failed_checks.append(
                f"slope error {slope_error:.4f} > {self.sigma_tol} * sigma ({fitted_slope.sigma:.4f})",
            )
        if (fitted_intercept.sigma is not None) and (
            intercept_error > self.sigma_tol * fitted_intercept.sigma
        ):
            failed_checks.append(
                f"intercept error {intercept_error:.4f} > {self.sigma_tol} * sigma"
                f" ({fitted_intercept.sigma:.4f})",
            )
        return failed_checks


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    test = TestLinearFit()
    test.run()

## } V-TEST
