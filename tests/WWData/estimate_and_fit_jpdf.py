## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import ComputePDFs, FitData
from Loki.WWPlots import PlotUtils


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def sampleFromEllipse(num_samples):
  center_x = 0
  center_y = 10
  semi_major_axis = 10
  semi_minor_axis = 2
  angle_deg = 90 / 2
  angle_rad = angle_deg * numpy.pi / 180
  x_samples = numpy.random.normal(0, semi_major_axis, int(num_samples))
  y_samples = numpy.random.normal(0, semi_minor_axis, int(num_samples))
  x_rotated = center_x + x_samples * numpy.cos(angle_rad) - y_samples * numpy.sin(angle_rad)
  y_rotated = center_y + x_samples * numpy.sin(angle_rad) + y_samples * numpy.cos(angle_rad)
  slope = numpy.tan(angle_rad)
  intercept = center_y - slope * center_x
  print(f"true trend: y = {slope:.3f} x + {intercept:.3f}")
  return x_rotated, y_rotated


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_points = 1e5
  num_bins   = 1e2
  x_samples, y_samples = sampleFromEllipse(num_points)
  bedges_rows, bedges_cols, jpdf = ComputePDFs.computeJPDF(
    data_x   = x_samples,
    data_y   = y_samples,
    num_bins = num_bins,
    smoothing_length = 2.0
  )
  fig, ax = PlotUtils.initFigure(num_cols=1)
  ax.contourf(bedges_rows[:-1], bedges_cols[:-1], jpdf.T, levels=20, cmap="Blues")
  ax.scatter(x_samples, y_samples, color="red", s=3, alpha=1e-2)
  # x_range = numpy.linspace(numpy.min(x_samples), numpy.max(x_samples), 100)
  # intercept, slope = FitData.fitLineToMasked2DJPDF(bedges_cols, bedges_rows, jpdf, percent_threshold=0.85, ax=ax)
  # intercept = intercept / 1.75
  # print(f"estimated fit: y = {slope:.3f} x + {intercept:.3f}")
  # ax.plot(x_range, x_range/slope - intercept, ls="--", color="black", zorder=7)
  ax.axhline(y=0, color="black", ls="--", zorder=1)
  ax.axvline(x=0, color="black", ls="--", zorder=1)
  # ax.set_xlim([ numpy.min(bedges_cols), numpy.max(bedges_cols) ])
  # ax.set_ylim([ numpy.min(bedges_rows), numpy.max(bedges_rows) ])
  ax.grid(True, which="both", linestyle="--", linewidth=0.5)
  PlotUtils.saveFigure(fig, "estimate_and_fit_jpdf.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT