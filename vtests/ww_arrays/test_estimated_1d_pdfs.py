## { V-TEST

##
## === DEPENDENCIES
##

## third-party
import numpy

## local
from jormi import ww_lists
from jormi.ww_arrays import compute_array_stats
from jormi.ww_io import manage_io
from jormi.ww_plots import manage_plots

##
## === BINNING CONVERGENCE TEST
##


def main():
    ## parameters
    rng = numpy.random.default_rng(seed=42)
    num_samples = int(1e5)
    binning_to_test = [5, 10, 50, 100]
    integral_error_tol = 1e-2
    ## distributions to test: each is a different shape to stress the estimator
    pdfs_to_test = {
        "delta": rng.normal(loc=10, scale=1e-9, size=num_samples),
        "uniform": rng.uniform(low=0, high=1, size=num_samples),
        "normal": rng.normal(loc=0, scale=1, size=num_samples),
        "exponential": rng.exponential(scale=1, size=num_samples),
    }
    num_pdfs = len(pdfs_to_test)
    fig, axs_grid = manage_plots.create_figure(
        num_rows=num_pdfs,
        num_cols=1,
        y_spacing=0.25,
    )
    ## estimate PDF for each distribution at multiple bin counts
    pdfs_that_failed = []
    for pdf_index, (pdf_label, pdf_samples) in enumerate(pdfs_to_test.items()):
        ax = axs_grid[pdf_index, 0]
        failed_bins = []
        for num_bins in binning_to_test:
            result = compute_array_stats.estimate_pdf(
                values=pdf_samples,
                num_bins=num_bins,
                bin_range_percent=1.5,
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
            ## normalisation check: sum(pdf * dx) should be ~1
            ax.step(bin_centers, estimated_pdf, where="mid", lw=2, label=f"{num_bins} bins")
            bin_widths = numpy.diff(result.bin_edges)
            pdf_integral = numpy.sum(estimated_pdf * bin_widths)
            if abs(pdf_integral - 1.0) > integral_error_tol:
                failed_bins.append(num_bins)
        ax.text(0.95, 0.95, pdf_label, ha="right", va="top", transform=ax.transAxes)
        ax.set_ylabel(r"PDF$(x)$")
        if failed_bins:
            print(f"Failed: {pdf_label} - integral out of tolerance for bins: {failed_bins}")
            pdfs_that_failed.append(pdf_label)
        else:
            print(f"Passed: {pdf_label}")
    axs_grid[-1, 0].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
    axs_grid[-1, 0].set_xlabel(r"$x$")
    ## save figure always so it can be inspected on failure
    file_dir = manage_io.get_caller_directory()
    fig_name = "estimated_1d_pdfs.png"
    fig_path = file_dir / fig_name
    manage_plots.save_figure(fig, fig_path)
    assert len(pdfs_that_failed) == 0, (
        f"Test failed for the following distributions: {ww_lists.as_string(pdfs_that_failed)}"
    )
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } V-TEST
