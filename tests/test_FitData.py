## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWStats import ComputePDFs
from Loki.WWPlots import PlotUtils


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def generate_ellipse_samples(num_samples):
  center_x, center_y = 0, 0
  a, b = 4, 2
  theta = numpy.pi / 4
  x_samples = numpy.random.normal(center_x, a, int(num_samples))
  y_samples = numpy.random.normal(center_y, b, int(num_samples))
  x_rot = (x_samples - center_x) * numpy.cos(theta) - (y_samples - center_y) * numpy.sin(theta) + center_x
  y_rot = (x_samples - center_x) * numpy.sin(theta) + (y_samples - center_y) * numpy.cos(theta) + center_y
  return x_rot, y_rot


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_points = 1e5
  num_bins   = 100
  x_samples, y_samples = generate_ellipse_samples(num_points)
  bedges_x, bedges_y, jpdf = ComputePDFs.computeJPDF(x_samples, y_samples, num_bins=num_bins)
  fig, ax = PlotUtils.initFigure(num_cols=1)
  ax.contourf(bedges_x[:-1], bedges_y[:-1], jpdf.T, levels=20, cmap="Blues")
  # ax.scatter(x_samples, y_samples, color="red", s=10, alpha=1e-1)
  ax.set_xlim([ numpy.min(bedges_x), numpy.max(bedges_x) ])
  ax.set_ylim([ numpy.min(bedges_y), numpy.max(bedges_y) ])
  PlotUtils.saveFigure(fig, "test_FitData.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT