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
  label = f"true: $y = {slope:.3f} x {str_sign} {numpy.abs(intercept):.3f}$"
  if ax is not None: ax.text(0.05, 0.95, label, ha="left", va="top", transform=ax.transAxes, fontsize=20)
  return x_rotated, y_rotated

def findMedianPoints(jpdf):
  p50_indices_grouped_rows = []
  for row_index in range(jpdf.shape[0]):
    row_slice = jpdf[row_index, :]
    row_slice_normalized = row_slice / numpy.sum(row_slice)
    cdf = numpy.cumsum(row_slice_normalized)
    idx = numpy.where(cdf >= 0.5)[0]
    if len(idx) > 0:
      p50_indices_grouped_rows.append(idx[0])
    else: p50_indices_grouped_rows.append(numpy.nan)
  p50_indices_grouped_cols = []
  for col_index in range(jpdf.shape[1]):
    col_slice = jpdf[:, col_index]
    col_slice_normalized = col_slice / numpy.sum(col_slice)
    cdf = numpy.cumsum(col_slice_normalized)
    idx = numpy.where(cdf >= 0.5)[0]
    if len(idx) > 0:
      p50_indices_grouped_cols.append(idx[0])
    else: p50_indices_grouped_cols.append(numpy.nan)
  return numpy.array(p50_indices_grouped_rows), numpy.array(p50_indices_grouped_cols)

def estimateTrendFromSymmetricBinnedData(bedges_rows, bedges_cols, jpdf, ax=None):
  index_of_median_grouped_row, index_of_median_grouped_col = findMedianPoints(jpdf)
  median_y_value_for_each_x = [ bedges_rows[index] for index in index_of_median_grouped_row if not numpy.isnan(index) ]
  median_x_value_for_each_y = [ bedges_cols[index] for index in index_of_median_grouped_col if not numpy.isnan(index) ]
  slope_from_x, intercept_from_x = numpy.polyfit(bedges_cols[:-1], median_y_value_for_each_x, 1)
  slope_from_y, intercept_from_y = numpy.polyfit(median_x_value_for_each_y, bedges_rows[:-1], 1)
  avg_slope     = 0.5 * (slope_from_x + slope_from_y)
  avg_intercept = 0.5 * (intercept_from_x + intercept_from_y)
  if ax is not None:
    ax.scatter(median_x_value_for_each_y, bedges_rows[:-1], color="red", s=10, zorder=5)
    ax.scatter(bedges_cols[:-1], median_y_value_for_each_x, color="red", s=10, zorder=5)
  return avg_slope, avg_intercept


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_points = 1e5
  num_bins   = 1e2
  bool_plot_samples = 0
  fig, ax = PlotUtils.initFigure()
  x_samples, y_samples = sampleFromEllipse(num_points, ax)
  bedges_rows, bedges_cols, jpdf = ComputePDFs.computeJPDF(
    data_x   = x_samples,
    data_y   = y_samples,
    num_bins = num_bins,
    smoothing_length = 2.0
  )
  ## assuming uniform binning
  bin_width_x = bedges_cols[1] - bedges_cols[0]
  bin_width_y = bedges_rows[1] - bedges_rows[0]
  integral = numpy.sum(jpdf * bin_width_x * bin_width_y)
  print(f"Integral of JPDF estimated with {num_bins:.0f} x {num_bins:.0f} bins: {integral:.3f}")
  ax.contourf(bedges_cols[:-1], bedges_rows[:-1], jpdf, levels=20, cmap="Blues")
  if bool_plot_samples: ax.scatter(x_samples, y_samples, color="red", s=3, alpha=1e-2)
  ## estimate trend of the distribution
  avg_slope, avg_intercept = estimateTrendFromSymmetricBinnedData(bedges_rows, bedges_cols, jpdf, ax)
  x_range = numpy.linspace(min(bedges_cols), max(bedges_cols), 100)
  ax.plot(x_range, avg_intercept + avg_slope * x_range, ls="--", color="black", zorder=7)
  str_sign = "-" if (avg_intercept < 0) else "+"
  label = f"estimated: $y = {avg_slope:.3f} x {str_sign} {numpy.abs(avg_intercept):.3f}$"
  ax.text(0.05, 0.85, label, ha="left", va="top", transform=ax.transAxes, fontsize=20)
  ## add annotations
  ax.set_xlabel(r"$x$")
  ax.set_ylabel(r"$y$")
  ax.axhline(y=0, color="black", ls="--", zorder=1)
  ax.axvline(x=0, color="black", ls="--", zorder=1)
  ax.set_xlim([ numpy.min(bedges_cols), numpy.max(bedges_cols) ])
  ax.set_ylim([ numpy.min(bedges_rows), numpy.max(bedges_rows) ])
  PlotUtils.saveFigure(fig, "estimate_and_fit_jpdf.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT