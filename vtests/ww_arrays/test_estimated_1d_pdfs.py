## { V-TEST

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_log
from jormi.ww_plots import annotate_panel, manage_figure, style_plots
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##


@dataclass(frozen=True)
class PDFScenario:
    label: str
    samples: numpy.ndarray[Any, numpy.dtype[Any]]


##
## === BINNING CONVERGENCE TEST
##


class TestEstimated1DPDFs:
    """
    Estimate PDFs of several known distributions at a range of bin counts and
    check that each estimate integrates to unity within `integral_error_tol`.
    """

    def __init__(
        self,
    ):
        ## sample parameters
        self.seed: int = 42
        self.num_samples: int = int(1e5)
        ## bin counts to test: stresses the estimator across resolutions
        self.bin_counts_to_test: list[int] = [5, 10, 50, 100]
        ## pass criterion: estimated PDF must integrate to ~1
        self.integral_error_tol: float = 1e-2

    def run(
        self,
    ) -> None:
        pdf_scenarios = self._generate_pdf_samples()
        num_pdfs = len(pdf_scenarios)
        figure, panels_grid = manage_figure.create_figure(
            num_rows=num_pdfs,
            num_cols=1,
            y_spacing=0.25,
        )
        failed_pdfs: list[str] = []
        for pdf_index, pdf_scenario in enumerate(pdf_scenarios):
            failed_bins = self._plot_and_check_pdf(
                panel=panels_grid[pdf_index, 0],
                pdf_samples=pdf_scenario.samples,
                pdf_label=pdf_scenario.label,
            )
            if failed_bins:
                manage_log.log_outcome(
                    text=f"{pdf_scenario.label} integral out of tolerance for bins: {failed_bins}",
                    outcome=manage_log.ActionOutcome.FAILURE,
                )
                failed_pdfs.append(pdf_scenario.label)
            else:
                manage_log.log_outcome(
                    text=f"{pdf_scenario.label}",
                    outcome=manage_log.ActionOutcome.SUCCESS,
                )
        panels_grid[-1, 0].legend(
            loc="upper right",
            bbox_to_anchor=(1, 0.9),
            fontsize=20,
        )
        panels_grid[-1, 0].set_xlabel(r"$x$")
        ## always save even on failure
        figure_path = Path(__file__).parent / "estimated_1d_pdfs.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=figure_path,
        )
        assert not failed_pdfs, (
            f"Test failed for the following distributions: {ww_lists.as_string(elems=failed_pdfs)}"
        )
        manage_log.log_action(
            title="Estimate 1D PDFs",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="All checks passed.",
        )

    def _generate_pdf_samples(
        self,
    ) -> list[PDFScenario]:
        rng = numpy.random.default_rng(seed=self.seed)
        ## each distribution is a different shape to stress the estimator
        return [
            PDFScenario(
                label="delta",
                samples=rng.normal(
                    loc=10,
                    scale=1e-9,
                    size=self.num_samples,
                ),
            ),
            PDFScenario(
                label="uniform",
                samples=rng.uniform(
                    low=0,
                    high=1,
                    size=self.num_samples,
                ),
            ),
            PDFScenario(
                label="normal",
                samples=rng.normal(
                    loc=0,
                    scale=1,
                    size=self.num_samples,
                ),
            ),
            PDFScenario(
                label="exponential",
                samples=rng.exponential(
                    scale=1,
                    size=self.num_samples,
                ),
            ),
        ]

    def _plot_and_check_pdf(
        self,
        *,
        panel: manage_figure.PlotPanel,
        pdf_samples: numpy.ndarray[Any, numpy.dtype[Any]],
        pdf_label: str,
    ) -> list[int]:
        failed_bins: list[int] = []
        for num_bins in self.bin_counts_to_test:
            result = compute_array_stats.estimate_pdf(
                values=pdf_samples,
                num_bins=num_bins,
            )
            bin_centers = result.bin_centers
            estimated_pdf = result.densities
            ## shape checks
            assert len(bin_centers) >= 3, (
                f"{pdf_label} ({num_bins} bins): expected at least 3 bins, got {len(bin_centers)}"
            )
            assert bin_centers.shape == estimated_pdf.shape, (
                f"{pdf_label} ({num_bins} bins): shape mismatch, "
                f"centers={bin_centers.shape}, pdf={estimated_pdf.shape}"
            )
            if len(bin_centers) > 3:
                assert len(bin_centers) == num_bins, (
                    f"{pdf_label}: expected {num_bins} centers, got {len(bin_centers)}"
                )
            panel.step(
                bin_centers,
                estimated_pdf,
                where="mid",
                lw=2,
                label=f"{num_bins} bins",
            )
            ## normalisation check: sum(pdf * dx) should be ~1
            bin_widths = numpy.diff(result.bin_edges)
            pdf_integral = numpy.sum(estimated_pdf * bin_widths)
            if abs(pdf_integral - 1.0) > self.integral_error_tol:
                failed_bins.append(num_bins)
        annotate_panel.add_text(
            panel=panel,
            x_pos=0.95,
            y_pos=0.95,
            label=pdf_label,
            x_alignment=box_positions.Positions.Side.Right,
            y_alignment=box_positions.Positions.Side.Top,
        )
        panel.set_ylabel(r"PDF$(x)$")
        return failed_bins


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_plots.set_theme()
    test = TestEstimated1DPDFs()
    test.run()

## } V-TEST
