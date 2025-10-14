## { TEST

##
## === DEPENDENCIES
##

import numpy
from jormi.utils import list_utils
from jormi.ww_io import io_manager
from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager

##
## === BINNING CONVERGENCE TEST
##


def main():
    num_samples = int(1e5)
    num_bins_to_test = [5, 10, 50, 100]
    pdfs_to_test = {
        "delta": numpy.random.normal(loc=10, scale=1e-9, size=num_samples),
        "uniform": numpy.random.uniform(low=0, high=1, size=num_samples),
        "normal": numpy.random.normal(loc=0, scale=1, size=num_samples),
        "exponential": numpy.random.exponential(scale=1, size=num_samples),
    }
    integral_tolerance = 1e-2
    num_pdfs = len(pdfs_to_test)
    fig, axs_grid = plot_manager.create_figure(num_rows=num_pdfs, y_spacing=0.25)
    if num_pdfs == 1: axs_grid = list(axs_grid)
    pdfs_that_failed = []
    for pdf_index, (pdf_label, pdf_samples) in enumerate(pdfs_to_test.items()):
        ax = axs_grid[pdf_index]
        for num_bins in num_bins_to_test:
            result = compute_stats.estimate_pdf(
                pdf_samples,
                num_bins=num_bins,
                bin_range_percent=1.5,
            )
            bin_centers = result.bin_centers
            estimated_pdf = result.density
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
            ax.step(bin_centers, estimated_pdf, where="mid", lw=2, label=f"{num_bins} bins")
            bin_widths = numpy.diff(result.bin_edges)
            pdf_integral = numpy.sum(estimated_pdf * bin_widths)
            if abs(pdf_integral - 1.0) > integral_tolerance: pdfs_that_failed.append(pdf_label)
        ax.text(0.95, 0.95, pdf_label, ha="right", va="top", transform=ax.transAxes)
        ax.set_ylabel(r"PDF$(x)$")
    axs_grid[-1].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
    axs_grid[-1].set_xlabel(r"$x$")
    directory = io_manager.get_caller_directory()
    file_name = "estimated_1d_pdfs.png"
    file_path = io_manager.combine_file_path_parts([directory, file_name])
    plot_manager.save_figure(fig, file_path)
    assert len(
        pdfs_that_failed,
    ) == 0, f"Test failed for the following methods: {list_utils.cast_to_string(pdfs_that_failed)}"
    print("All tests passed successfully!")


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } TEST
