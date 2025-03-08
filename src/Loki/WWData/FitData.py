import numpy as np
from scipy.stats import mstats

def fitLineToMasked2DJPDF(bedges_rows, bedges_cols, jpdf, percent_threshold, ax=None):
  jpdf = np.array(jpdf)
  level_threshold = percent_threshold * np.max(jpdf)
  mask = jpdf > level_threshold
  mg_rows, mg_cols = np.meshgrid(bedges_rows, bedges_cols, indexing="ij")
  x = mg_rows[1:, 1:][mask].flatten()
  y = mg_cols[1:, 1:][mask].flatten()
  slope = mstats.theilslopes(y, x)[0]
  intercept = np.median(y - slope * x)
  if ax is not None:
    mg_rows_trimmed = mg_rows[:-1, :-1]
    mg_cols_trimmed = mg_cols[:-1, :-1]
    ax.contour(
      mg_cols_trimmed, mg_rows_trimmed, jpdf,
      levels=[level_threshold], colors="r", linewidths=1.5, zorder=5
    )
  return intercept, slope
