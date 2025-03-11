## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import ComputePDFs
from Loki.WWPlots import PlotUtils


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def sampleFromEllipse(num_samples, ax=None):
  center_x = 30
  center_y = 100
  semi_major_axis = 10
  semi_minor_axis = 3
  angle_deg = 90 / 2
  angle_rad = angle_deg * numpy.pi / 180
  x_samples = numpy.random.normal(0, semi_major_axis, int(num_samples))
  y_samples = numpy.random.normal(0, semi_minor_axis, int(num_samples))
  x_rotated = center_x + x_samples * numpy.cos(angle_rad) - y_samples * numpy.sin(angle_rad)
  y_rotated = center_y + x_samples * numpy.sin(angle_rad) + y_samples * numpy.cos(angle_rad)
  slope = numpy.tan(angle_rad)
  intercept = center_y - slope * center_x
  str_sign = "-" if (intercept < 0) else "+"
  label = f"true trend: $y = {slope:.3f} x {str_sign} {numpy.abs(intercept):.3f}$"
  if ax is not None: ax.text(0.05, 0.95, label, ha="left", va="top", transform=ax.transAxes, fontsize=20)
  return x_rotated, y_rotated


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_points         = 1e5
  num_bins           = 1e2
  bool_plot_samples  = False
  integral_tolerance = 1e-2
  fig, ax = PlotUtils.initFigure()
  x_samples, y_samples = sampleFromEllipse(num_points, ax)
  bedges_rows, bedges_cols, jpdf = ComputePDFs.compute2DJPDF(
    data_x   = x_samples,
    data_y   = y_samples,
    num_bins = num_bins,
    smoothing_length = 2.0
  )
  ## assuming uniform binning
  bin_width_x = bedges_cols[1] - bedges_cols[0]
  bin_width_y = bedges_rows[1] - bedges_rows[0]
  pdf_integral = numpy.sum(jpdf * bin_width_x * bin_width_y)
  ax.contourf(bedges_cols[:-1], bedges_rows[:-1], jpdf, levels=20, cmap="Blues")
  if bool_plot_samples: ax.scatter(x_samples, y_samples, color="red", s=3, alpha=1e-2)
  ## add annotations
  ax.set_xlabel(r"$x$")
  ax.set_ylabel(r"$y$")
  ax.axhline(y=0, color="black", ls="--", zorder=1)
  ax.axvline(x=0, color="black", ls="--", zorder=1)
  ax.set_xlim([ numpy.min(bedges_cols), numpy.max(bedges_cols) ])
  ax.set_ylim([ numpy.min(bedges_rows), numpy.max(bedges_rows) ])
  PlotUtils.saveFigure(fig, "estimated_2d_jpdf.png")
  assert abs(pdf_integral - 1.0) < integral_tolerance, f"Test failed: JPDF with {num_bins} x {num_bins} bins sums to {pdf_integral:.6f}"
  print("Test passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT