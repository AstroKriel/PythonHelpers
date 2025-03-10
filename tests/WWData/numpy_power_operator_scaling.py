## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy as np
from Loki.WWPlots import PlotUtils, PlotAnnotations


## ###############################################################
## COMPARE EXPONENT OPERATOR VS NUMPY.POWER PERFORMANCE
## ###############################################################
def main():
  exponent = 2.5
  fig, ax = PlotUtils.initFigure(fig_aspect_ratio=(5,6))
  for num_dims in [ 1, 2, 3 ]:
    list_sizes            = []
    list_timings_operator = []
    list_timings_power    = []
    if   num_dims == 1: marker = "o"
    elif num_dims == 2: marker = "s"
    elif num_dims == 3: marker = "D"
    else: raise ValueError("Unsuported array dimensions.")
    for expected_num_elems in [ 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 ]:
      if num_dims == 1:
        shape = (int(expected_num_elems),)
      elif num_dims == 2:
        side_length = int(np.sqrt(expected_num_elems))
        shape = (side_length, side_length)
      elif num_dims == 3:
        side_length = int(round(expected_num_elems ** (1/3)))
        shape = (side_length, side_length, side_length)
      else: raise ValueError("Unsuported array dimensions.")
      random_array = np.random.rand(*shape)
      actual_elements = random_array.size
      list_sizes.append(actual_elements)
      ## measure time for `a**b`
      start_time = time.perf_counter()
      _ = random_array ** exponent
      end_time = time.perf_counter()
      list_timings_operator.append(end_time - start_time)
      ## measure time for `numpy.power(a,b)`
      start_time = time.perf_counter()
      _ = np.power(random_array, exponent)
      end_time = time.perf_counter()
      list_timings_power.append(end_time - start_time)
    ax.plot(list_sizes, list_timings_operator, color="blue", ls="-", marker=marker)
    ax.plot(list_sizes, list_timings_power, color="red", ls=":", marker=marker)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Total number of elements")
  ax.set_ylabel("Execution time (s)")
  PlotAnnotations.addLegend_fromArtists(
    ax                 = ax,
    list_artists       = [ "-", ":", "o", "s", "D" ],
    list_legend_labels = [ "a**b", "numpy.power(a,b)", "1D array", "2D array", "3D array" ],
    list_marker_colors = [ "blue", "red", "black", "black", "black" ],
    label_color        = "black",
    loc                = "upper left",
    bbox               = (0.0, 1.0),
  )
  ax.grid(True, which="major", linestyle="--", linewidth=0.5)
  PlotUtils.saveFigure(fig, "numpy_power_operator_scaling.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST