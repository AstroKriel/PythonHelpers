## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from jormungandr.utils import list_utils
from jormungandr.ww_io import file_manager


## ###############################################################
## FUNCTIONS
## ###############################################################
def read_vi_data(
  directory     : str,
  file_name     : str = "Turb.dat",
  dataset_name  : str | None = None,
  dataset_index : int | None = None,
  time_norm     : float = 1.0,
  time_start    : float = 0.0,
  time_end      : float | None = None,
  debug         : bool = False,
  verbose       : bool = False,
  print_header  : bool = False,
) -> tuple[list[float], list[float]]:
  def _print_header(header):
    print(f"Available datasets in: {file_path}")
    for _dataset_index, _dataset_name in enumerate(header.strip().split()):
      print(f"\t> {_dataset_index:3d}: {_dataset_name}")
  file_path = file_manager.create_file_path([directory, file_name])
  file_manager.does_file_exist(file_path=file_path, raise_error=True)
  with open(file_path, "r") as file:
    file_lines = file.readlines()
  header = file_lines[0].split()
  is_new_format = "#01_time" in header
  if not(is_new_format): print("looking at an old dataset.")
  lookup_dataset_index = {
    "kin"  : 9  if is_new_format else 6,
    "mag"  : 11 if is_new_format else 29,
    "mach" : 13 if is_new_format else 8,
  }
  if print_header:
    _print_header(header)
    return [], []
  if dataset_index is None:
    if dataset_name is None: raise ValueError("Error: You need to either provide `dataset_index` or `dataset_name`.")
    dataset_name = dataset_name.lower()
    if dataset_name not in lookup_dataset_index:
      valid_datasets = list_utils.cast_to_string(lookup_dataset_index.keys())
      _print_header(header)
      raise ValueError(f"Error: `{dataset_name}` is an invalid dataset. Choose from: {valid_datasets}, or provide `dataset_index` directly.")
    dataset_index = lookup_dataset_index[dataset_name]
  time_index   = 0 # time-values should always be stored in the first column-index
  prev_time    = numpy.inf
  num_datasets = len(file_lines[0].split()) # each header entry is a quantity
  times        = []
  values       = []
  ## read backwards to prioritise the latest instance of a data-point (e.g., when the sim has been restarted)
  for line in reversed(file_lines[1:]):
    datasets = line.strip().split()
    if len(datasets) != num_datasets: continue
    if "#" in datasets[time_index] or "#" in datasets[dataset_index]: continue
    try:
      time_value    = float(datasets[time_index]) / time_norm
      dataset_value = float(datasets[dataset_index])
    except ValueError: continue
    ## only add data at times that maintain monotonicity in time values
    if time_value < prev_time:
      if (dataset_value == 0.0) and (time_value > 0):
        message = (f"{file_name}: field[{dataset_index:d}] = 0.0 at time = {time_value:.3f}")
        if debug: raise ValueError(f"Error: {message}")
        if verbose: print(f"Warning: {message}")
        continue
      times.append(time_value)
      values.append(dataset_value)
      prev_time = time_value
  times.reverse()
  values.reverse()
  time_end    = time_end if time_end is not None else times[-1]
  start_index = list_utils.get_index_of_closest_value(times, time_start)
  end_index   = list_utils.get_index_of_closest_value(times, time_end)
  if start_index == end_index: end_index = min(end_index+1, len(times)) # avoid empty ranges
  subsetted_times  = numpy.array(times[start_index:end_index])
  subsetted_values = numpy.array(values[start_index:end_index])
  return subsetted_times, subsetted_values


## END OF MODULE