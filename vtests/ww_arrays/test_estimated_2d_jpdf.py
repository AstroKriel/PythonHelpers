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
from jormi.ww_plots import manage_plots, style_plots

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
## === JPDF NORMALISATION TEST
##


class TestEstimated2DJPDF:
    """
    Estimate the 2D JPDF of a rotated elliptical Gaussian sample and check that it
    integrates to unity: abs(integral - 1) must stay below `integral_error_tol`.
    """

    def __init__(
        self,
    ):
        ## sample parameters
        self.seed: int = 42
        self.num_points: int = int(3e5)
        ## estimator parameters
        self.num_bins: int = int(1e2)
        self.smoothing_length: float = 2.0
        ## overlay raw samples on the JPDF for debugging
        self.plot_samples: bool = False
        ## pass criterion: JPDF integral must match unity within this tolerance
        self.integral_error_tol: float = 1e-2

    def run(
        self,
    ) -> None:
        rng = numpy.random.default_rng(seed=self.seed)
        x_samples, y_samples = sample_from_ellipse(
            num_samples=self.num_points,
            rng=rng,
        )
        fig, ax = manage_plots.create_figure()
        result = compute_array_stats.estimate_jpdf(
            data_x=x_samples,
            data_y=y_samples,
            num_bins=self.num_bins,
            smoothing_length=self.smoothing_length,
        )
        jpdf = result.densities
        bin_centers_rows = result.row_centers
        bin_centers_cols = result.col_centers
        self._plot_jpdf(
            ax=ax,
            jpdf=jpdf,
            bin_centers_rows=bin_centers_rows,
            bin_centers_cols=bin_centers_cols,
            x_samples=x_samples,
            y_samples=y_samples,
            plot_samples=self.plot_samples,
        )
        ## normalisation check: sum(jpdf * dA) should be ~1
        bin_widths_x = numpy.diff(result.col_edges)
        bin_widths_y = numpy.diff(result.row_edges)
        pdf_integral = numpy.sum(
            jpdf * bin_widths_y[:, numpy.newaxis] * bin_widths_x[numpy.newaxis, :]
        )
        ## always save even on failure
        fig_path = Path(__file__).parent / "estimated_2d_jpdf.png"
        manage_plots.save_figure(
            fig=fig,
            fig_path=fig_path,
        )
        assert abs(pdf_integral - 1.0) < self.integral_error_tol, (
            f"JPDF with {self.num_bins} x {self.num_bins} bins sums to {pdf_integral:.6f}"
        )
        manage_log.log_action(
            title="Estimate 2D JPDF",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
            notes={"integral": f"{pdf_integral:.6f}"},
        )

    def _plot_jpdf(
        self,
        *,
        ax: manage_plots.PlotAxis,
        jpdf: numpy.ndarray[Any, numpy.dtype[Any]],
        bin_centers_rows: numpy.ndarray[Any, numpy.dtype[Any]],
        bin_centers_cols: numpy.ndarray[Any, numpy.dtype[Any]],
        x_samples: numpy.ndarray[Any, numpy.dtype[Any]],
        y_samples: numpy.ndarray[Any, numpy.dtype[Any]],
        plot_samples: bool,
    ) -> None:
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


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    test = TestEstimated2DJPDF()
    test.run()

## } V-TEST
