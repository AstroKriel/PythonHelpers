## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWStats import ComputePDFs
from Loki.WWPlots import PlotUtils


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def sampleFromEllipse(num_samples):
  center_x = 0
  center_y = 0
  semi_major_axis = 4
  semi_minor_axis = 2
  rotation_angle_rad = numpy.pi / 4
  x_samples = numpy.random.normal(center_x, semi_major_axis, int(num_samples))
  y_samples = numpy.random.normal(center_y, semi_minor_axis, int(num_samples))
  x_rot = (x_samples - center_x) * numpy.cos(rotation_angle_rad) - (y_samples - center_y) * numpy.sin(rotation_angle_rad) + center_x
  y_rot = (x_samples - center_x) * numpy.sin(rotation_angle_rad) + (y_samples - center_y) * numpy.cos(rotation_angle_rad) + center_y
  return x_rot, y_rot


## ###############################################################
## BINNING CONVERGENCE TEST
## ###############################################################
def main():
  num_points = 1e5
  num_bins   = 1e2
  x_samples, y_samples = sampleFromEllipse(num_points)
  bedges_x, bedges_y, jpdf = ComputePDFs.computeJPDF(x_samples, y_samples, num_bins=num_bins, smoothing_length=2.0)
  fig, ax = PlotUtils.initFigure(num_cols=1)
  ax.contourf(bedges_x[:-1], bedges_y[:-1], jpdf.T, levels=20, cmap="Blues")
  ax.scatter(x_samples, y_samples, color="red", s=3, alpha=1e-2)
  ax.set_xlim([ numpy.min(bedges_x), numpy.max(bedges_x) ])
  ax.set_ylim([ numpy.min(bedges_y), numpy.max(bedges_y) ])
  PlotUtils.saveFigure(fig, "test_ComputePDFs_computeJPDF.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF SCRIPT