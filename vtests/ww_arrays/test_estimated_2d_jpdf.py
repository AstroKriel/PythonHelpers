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
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_plots

##
## === HELPER FUNCTIONS
##


def sample_from_ellipse(
    num_samples: int,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray[Any, numpy.dtype[Any]], numpy.ndarray[Any, numpy.dtype[Any]]]:
    x_center = 30
    y_center = 100
    semi_major_axis = 10
    semi_minor_axis = 3
    angle_deg = 90 / 2
    angle_rad = angle_deg * numpy.pi / 180
    x_samples = rng.normal(0, semi_major_axis, int(num_samples))
    y_samples = rng.normal(0, semi_minor_axis, int(num_samples))
    x_rotated = x_center + x_samples * numpy.cos(angle_rad) - y_samples * numpy.sin(angle_rad)
    y_rotated = y_center + x_samples * numpy.sin(angle_rad) + y_samples * numpy.cos(angle_rad)
    return x_rotated, y_rotated


##
## === BINNING CONVERGENCE TEST
##


def main():
    ## parameters
    num_points = int(3e5)
    num_bins = int(1e2)
    plot_samples = False  # set True to overlay raw samples on the JPDF for debugging
    integral_error_tol = 1e-2
    ## sample data: rotated elliptical Gaussian
    rng = numpy.random.default_rng(seed=42)
    fig, ax = manage_plots.create_figure()
    x_samples, y_samples = sample_from_ellipse(num_points, rng)
    ## estimate JPDF
    result = compute_array_stats.estimate_jpdf(
        data_x=x_samples,
        data_y=y_samples,
        num_bins=num_bins,
        smoothing_length=2.0,
    )
    ## compute integral for normalisation check: sum(jpdf * dA) should be ~1
    bin_centers_rows = result.row_centers
    bin_centers_cols = result.col_centers
    jpdf = result.densities
    bin_widths_x = numpy.diff(result.col_edges)
    bin_widths_y = numpy.diff(result.row_edges)
    pdf_integral = numpy.sum(jpdf * bin_widths_y[:, numpy.newaxis] * bin_widths_x[numpy.newaxis, :])
    ## plot JPDF (always saved so it can be inspected on failure)
    ax.imshow(
        jpdf,
        extent=(
            bin_centers_cols.min(),
            bin_centers_cols.max(),
            bin_centers_rows.min(),
            bin_centers_rows.max(),
        ),
        origin="lower",
        aspect="auto",
        cmap="Blues",
    )
    if plot_samples:
        ax.scatter(
            x_samples,
            y_samples,
            color="red",
            s=3,
            alpha=1e-2,
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.axhline(
        y=0.0,
        color="black",
        ls="--",
        zorder=1,
    )
    ax.axvline(
        x=0.0,
        color="black",
        ls="--",
        zorder=1,
    )
    ax.set_xlim((numpy.min(bin_centers_cols), numpy.max(bin_centers_cols)))
    ax.set_ylim((numpy.min(bin_centers_rows), numpy.max(bin_centers_rows)))
    fig_name = "estimated_2d_jpdf.png"
    fig_path = Path(__file__).parent / fig_name
    manage_plots.save_figure(
        fig=fig,
        fig_path=fig_path,
    )
    ## check
    assert abs(pdf_integral - 1.0) < integral_error_tol, (
        f"Test failed: JPDF with {num_bins} x {num_bins} bins sums to {pdf_integral:.6f}"
    )
    manage_log.log_action(
        title="Estimate 2D JPDF",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Test passed successfully.",
        notes={"integral": f"{pdf_integral:.6f}"},
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } V-TEST
