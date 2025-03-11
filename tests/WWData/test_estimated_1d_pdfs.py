## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import ComputePDFs
from Loki.WWPlots import PlotUtils


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_samples = int(1e5)
  list_bin_sizes = [ 5, 10, 50, 100 ]
  dict_pdfs = {
    "delta"       : numpy.random.normal(loc=10, scale=1e-9, size=num_samples),
    "uniform"     : numpy.random.uniform(low=0, high=1, size=num_samples),
    "normal"      : numpy.random.normal(loc=0, scale=1, size=num_samples),
    "exponential" : numpy.random.exponential(scale=1, size=num_samples),
  }
  integral_tolerance = 1e-2
  num_pdfs = len(dict_pdfs)
  fig, axs = PlotUtils.initFigure(num_rows=num_pdfs)
  if num_pdfs == 1: axs = list(axs)
  list_failed_methods = []
  for pdf_index, (label, data) in enumerate(dict_pdfs.items()):
    ax = axs[pdf_index]
    for num_bins in list_bin_sizes:
      bedges, pdf = ComputePDFs.compute1DPDF(data, num_bins=num_bins, bedge_extend_factor=0.5)
      ax.step(bedges, pdf, where="pre", lw=2, label=f"{num_bins} bins")
      ## assuming uniform binning
      bin_widths = bedges[1] - bedges[0]
      pdf_integral = numpy.sum(pdf * bin_widths)
      if abs(pdf_integral - 1.0) > integral_tolerance: list_failed_methods.append(label)
    ax.text(0.95, 0.95, label, ha="right", va="top", transform=ax.transAxes)
    ax.set_ylabel(r"PDF$(x)$")
  axs[-1].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
  axs[-1].set_xlabel(r"$x$")
  PlotUtils.saveFigure(fig, "estimated_1d_pdfs.png")
  assert len(list_failed_methods) == 0, f"Test failed for the following methods: {list_failed_methods}"
  print("All tests passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST