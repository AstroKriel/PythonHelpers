## { TEST

##
## === DEPENDENCIES
##

## third-party
import numpy

## local
from jormi.ww_data import fit_series
from jormi.ww_data.series_types import GaussianSeries
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots

##
## === FIT ACCURACY TEST
##


def main():
    ## model parameters: y_values = true_slope * x_values + true_intercept
    true_slope = 2.5
    true_intercept = 1.0
    noise_sigma = 1.5
    num_points = 20
    num_sigma_tol = 3.0  # recovered params must lie within N sigma of truth
    rng = numpy.random.default_rng(seed=42)
    ## construct noisy linear dataset
    x_values = numpy.linspace(0.0, 10.0, num_points)
    y_sigmas = noise_sigma * numpy.ones_like(x_values)
    y_values = true_slope * x_values + true_intercept + rng.normal(scale=y_sigmas)
    series = GaussianSeries(
        x_values=x_values,
        y_values=y_values,
        y_sigmas=y_sigmas,
    )
    ## fit methods to test: free linear fit and fixed-slope fit
    fits_to_test = {
        "linear model": fit_series.fit_linear_model(series),
        "fixed slope": fit_series.fit_line_with_fixed_slope(
            gaussian_series=series,
            fixed_slope=true_slope,
        ),
    }
    ## plot: column stack, one row per fit method, shared x axis
    num_fits = len(fits_to_test)
    fig, axs_grid = manage_plots.create_figure(
        num_rows=num_fits,
        num_cols=1,
        share_x=True,
    )
    x_fit_values = numpy.linspace(series.x_bounds[0], series.x_bounds[1], 200)
    fits_that_failed = []
    for fit_index, (fit_label, fit) in enumerate(fits_to_test.items()):
        ax = axs_grid[fit_index, 0]
        is_top_ax = fit_index == 0
        is_bottom_ax = fit_index == num_fits - 1
        ## check: recovered params within N sigma of truth (skip if sigma unavailable)
        fitted_slope = fit.slope
        fitted_intercept = fit.intercept
        slope_error = abs(fitted_slope.value - true_slope)
        intercept_error = abs(fitted_intercept.value - true_intercept)
        failed_checks = []
        if (fitted_slope.sigma is not None) and (slope_error > num_sigma_tol * fitted_slope.sigma):
            failed_checks.append(
                f"slope error {slope_error:.4f} > {num_sigma_tol} * sigma ({fitted_slope.sigma:.4f})",
            )
        if (fitted_intercept.sigma is not None) and (intercept_error
                                                     > num_sigma_tol * fitted_intercept.sigma):
            failed_checks.append(
                f"intercept error {intercept_error:.4f} > {num_sigma_tol} * sigma ({fitted_intercept.sigma:.4f})",
            )
        ## plot data with errorbars and overlaid fit line
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_sigmas,
            fmt="o",
            color="black",
            label="data" if is_top_ax else None,
        )
        ax.plot(
            x_fit_values,
            fit.evaluate_fit(x_fit_values),
            color="red",
            label=fit_label,
        )
        ax.set_ylabel("y")
        ax.legend(
            fontsize=20,
            loc="upper left",
        )
        if is_bottom_ax:
            ax.set_xlabel("x")
        else:
            ax.tick_params(labelbottom=False)
        if failed_checks:
            for check_msg in failed_checks:
                print(f"Failed: {fit_label} - {check_msg}")
            fits_that_failed.append(fit_label)
        else:
            print(
                f"Passed: {fit_label}"
                f" (slope={fitted_slope.value:.4f}, intercept={fitted_intercept.value:.4f})",
            )
    ## save figure always so it can be inspected on failure
    file_dir = manage_io.get_caller_directory()
    fig_name = "linear_fit.png"
    fig_path = manage_io.combine_file_path_parts([file_dir, fig_name])
    manage_plots.save_figure(fig, fig_path)
    assert len(fits_that_failed) == 0, (f"Test failed for the following fit methods: {fits_that_failed}")
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } TEST
