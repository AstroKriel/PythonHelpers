## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWStats import ComputePDFs
from Loki.WWPlots import PlotUtils


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_samples = int(1e5)
  dict_pdfs = {
    "uniform"     : numpy.random.uniform(low=0, high=1, size=num_samples),
    "normal"      : numpy.random.normal(loc=0, scale=1, size=num_samples),
    "exponential" : numpy.random.exponential(scale=1, size=num_samples),
  }
  list_binning_options = [ 5, 10, 50, 100 ]
  num_pdfs = len(dict_pdfs)
  fig, axs = PlotUtils.initFigure(num_rows=num_pdfs)
  if num_pdfs == 1: axs = list(axs)
  for pdf_index, (label, data) in enumerate(dict_pdfs.items()):
    ax = axs[pdf_index]
    for num_bins in list_binning_options:
      bedges, pdf = ComputePDFs.compute1DPDF(data, num_bins=num_bins, bedge_extend_factor=0.5)
      ax.step(bedges, pdf, where="pre", lw=2, label=f"{num_bins} bins")
    ax.text(0.95, 0.95, label, ha="right", va="top", transform=ax.transAxes)
    ax.set_ylabel(r"PDF$(x)$")
  axs[-1].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
  axs[-1].set_xlabel(r"$x$")
  PlotUtils.saveFigure(fig, "test_Compute1DPDFs.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST