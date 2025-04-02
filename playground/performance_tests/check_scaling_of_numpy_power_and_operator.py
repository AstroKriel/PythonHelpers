## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy
from loki.ww_plots import plot_manager, PlotAnnotations


## ###############################################################
## COMPARE EXPONENT OPERATOR VS NUMPY.POWER PERFORMANCE
## ###############################################################
def main():
  num_repeats = 10
  exponent = 2.5
  fig, ax = plot_manager.create_figure(fig_aspect_ratio=(5,6))
  for num_dims in [ 3 ]:
    number_of_values = []
    ave_execution_time_of_operator = []
    std_execution_time_of_operator = []
    ave_execution_time_of_power    = []
    std_execution_time_of_power    = []
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
      true_number_of_values = len(random_array)
      number_of_values.append(true_number_of_values)
      execution_times_of_operator = []
      execution_times_of_power    = []
      for _ in range(num_repeats):
        ## measure execution time of `a**b`
        start_time = time.perf_counter()
        _ = random_array ** exponent
        end_time = time.perf_counter()
        execution_times_of_operator.append(end_time - start_time)
        ## measure execution time of `numpy.power(a,b)`
        start_time = time.perf_counter()
        _ = numpy.power(random_array, exponent)
        end_time = time.perf_counter()
        execution_times_of_power.append(end_time - start_time)
      ave_execution_time_of_operator.append(numpy.mean(execution_times_of_operator))
      std_execution_time_of_operator.append(numpy.std(execution_times_of_operator))
      ave_execution_time_of_power.append(numpy.mean(execution_times_of_power))
      std_execution_time_of_power.append(numpy.std(execution_times_of_power))
    ax.errorbar(
      number_of_values,
      ave_execution_time_of_operator,
      yerr = std_execution_time_of_operator,
      fmt=marker, capsize=7.5, color="blue", ls="-"
    )
    ax.errorbar(
      number_of_values,
      ave_execution_time_of_power,
      yerr = std_execution_time_of_power,
      fmt=marker, capsize=7.5, color="red", ls=":"
    )
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("Total number of elements")
  ax.set_ylabel("Execution time (seconds)")
  PlotAnnotations.add_custom_legend(
    ax           = ax,
    artists      = [ "-", ":", "o", "s", "D" ],
    labels       = [ "a**b", "numpy.power(a,b)", "1D array", "2D array", "3D array" ],
    colors       = [ "blue", "red", "black", "black", "black" ],
    text_color   = "black",
    position     = "upper left",
    anchor       = (0.0, 1.0),
    enable_frame = True
  )
  ax.grid(True, which="major", linestyle="--", linewidth=0.5)
  plot_manager.save_figure(fig, "scaling_of_numpy_power_and_operator.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST
