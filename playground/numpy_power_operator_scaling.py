## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy
from Loki.WWPlots import PlotUtils, PlotAnnotations


## ###############################################################
## COMPARE EXPONENT OPERATOR VS NUMPY.POWER PERFORMANCE
## ###############################################################
def main():
  num_repeats = 10
  exponent = 2.5
  fig, ax = PlotUtils.initFigure(fig_aspect_ratio=(5,6))
  for num_dims in [ 3 ]:
    list_sizes = []
    list_ave_timings_operator = []
    list_ave_timings_power    = []
    list_std_timings_operator = []
    list_std_timings_power    = []
    if   num_dims == 1: marker = "o"
    elif num_dims == 2: marker = "s"
    elif num_dims == 3: marker = "D"
    else: raise ValueError("Unsupported array dimensions.")
    for expected_num_elems in [ 10, 25, 50, 75, 1e2, 1e3, 1e4, 1e5, 1e6 ]:
      if num_dims == 1:
        shape = (int(expected_num_elems),)
      elif num_dims == 2:
        side_length = int(numpy.sqrt(expected_num_elems))
        shape = (side_length, side_length)
      elif num_dims == 3:
        side_length = int(round(expected_num_elems ** (1/3)))
        shape = (side_length, side_length, side_length)
      else: raise ValueError("Unsupported array dimensions.")
      random_array = numpy.random.rand(*shape)
      actual_num_elems = random_array.size
      list_sizes.append(actual_num_elems)
      timings_operator = []
      timings_power = []
      for _ in range(num_repeats):
        ## measure time for `a**b`
        start_time = time.perf_counter()
        _ = random_array ** exponent
        end_time = time.perf_counter()
        timings_operator.append(end_time - start_time)
        ## measure time for `numpy.power(a,b)`
        start_time = time.perf_counter()
        _ = numpy.power(random_array, exponent)
        end_time = time.perf_counter()
        timings_power.append(end_time - start_time)
      list_ave_timings_operator.append(numpy.mean(timings_operator))
      list_std_timings_operator.append(numpy.std(timings_operator))
      list_ave_timings_power.append(numpy.mean(timings_power))
      list_std_timings_power.append(numpy.std(timings_power))
    ax.errorbar(list_sizes, list_ave_timings_operator, yerr=list_std_timings_operator, fmt=marker, capsize=7.5, color="blue", ls="-")
    ax.errorbar(list_sizes, list_ave_timings_power,    yerr=list_std_timings_power,    fmt=marker, capsize=7.5, color="red", ls=":")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Total number of elements")
  ax.set_ylabel("Execution time (seconds)")
  PlotAnnotations.addCustomLegend(
    ax           = ax,
    list_artists = [ "-", ":", "o", "s", "D" ],
    list_labels  = [ "a**b", "numpy.power(a,b)", "1D array", "2D array", "3D array" ],
    list_colors  = [ "blue", "red", "black", "black", "black" ],
    label_color  = "black",
    loc          = "upper left",
    bbox         = (0.0, 1.0),
  )
  ax.grid(True, which="major", linestyle="--", linewidth=0.5)
  PlotUtils.saveFigure(fig, "numpy_power_operator_scaling.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST
