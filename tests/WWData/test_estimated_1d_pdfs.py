## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import ComputeStats
from Loki.WWPlots import PlotUtils


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_samples = int(1e5)
  num_bins_to_test = [ 5, 10, 50, 100 ]
  pdfs_to_test = {
    "delta"       : numpy.random.normal(loc=10, scale=1e-9, size=num_samples),
    "uniform"     : numpy.random.uniform(low=0, high=1, size=num_samples),
    "normal"      : numpy.random.normal(loc=0, scale=1, size=num_samples),
    "exponential" : numpy.random.exponential(scale=1, size=num_samples),
  }
  integral_tolerance = 1e-2
  num_pdfs = len(pdfs_to_test)
  fig, axs = PlotUtils.create_figure(num_rows=num_pdfs)
  if num_pdfs == 1: axs = list(axs)
  pdfs_that_failed = []
  for pdf_index, (pdf_label, pdf_samples) in enumerate(pdfs_to_test.items()):
    ax = axs[pdf_index]
    for num_bins in num_bins_to_test:
      bin_edges, estimated_pdf = ComputeStats.compute_pdf(pdf_samples, num_bins=num_bins, extend_bin_edge_percent=0.5)
      ax.step(bin_edges, estimated_pdf, where="pre", lw=2, label=f"{num_bins} bins")
      bin_width = bin_edges[1] - bin_edges[0] # assuming uniform binning
      pdf_integral = numpy.sum(estimated_pdf * bin_width)
      if abs(pdf_integral - 1.0) > integral_tolerance: pdfs_that_failed.append(pdf_label)
    ax.text(0.95, 0.95, pdf_label, ha="right", va="top", transform=ax.transAxes)
    ax.set_ylabel(r"PDF$(x)$")
  axs[-1].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
  axs[-1].set_xlabel(r"$x$")
  PlotUtils.save_figure(fig, "estimated_1d_pdfs.png")
  assert len(pdfs_that_failed) == 0, f"Test failed for the following methods: {pdfs_that_failed}"
  print("All tests passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST