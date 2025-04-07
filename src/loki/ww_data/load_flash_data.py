## START OF LIBRARY


## ###############################################################
## DEPENDENCIES
## ###############################################################
import h5py
import numpy
from loki.utils import list_utils


## ###############################################################
## FUNCTIONS FOR LOADING FLASH DATA
## ###############################################################
def reformat_flash_sfield_v1(
    sfield     : numpy.ndarray,
    num_blocks : tuple[int, int, int],
    num_procs  : tuple[int, int, int],
  ):
  xblocks, yblocks, zblocks = num_blocks
  iprocs, jprocs, kprocs = num_procs
  sfield_sorted = numpy.zeros(
    shape = (
      zblocks * kprocs,
      yblocks * jprocs,
      xblocks * iprocs,
    ),
    dtype = numpy.float32,
    order = "C"
  )
  ## [ num_cells_per_block, yblocks, zblocks, xblocks ] -> [ kprocs*zblocks, jprocs*yblocks, iprocs*xblocks ]
  block_index = 0
  for kproc_index in range(kprocs):
    for jproc_index in range(jprocs):
      for iproc_index in range(iprocs):
        sfield_sorted[
          kproc_index * zblocks : (kproc_index+1) * zblocks,
          jproc_index * yblocks : (jproc_index+1) * yblocks,
          iproc_index * xblocks : (iproc_index+1) * xblocks,
        ] = sfield[block_index, :, :, :]
        block_index += 1
  ## rearrange indices [ z, y, x ] -> [ x, y, z ]
  sfield_sorted = numpy.transpose(sfield_sorted, [2, 1, 0])
  return sfield_sorted

def reformat_flash_sfield_v2(
    sfield     : numpy.ndarray,
    num_blocks : tuple[int, int, int],
    num_procs  : tuple[int, int, int],
  ):
  xblocks, yblocks, zblocks = num_blocks
  iprocs, jprocs, kprocs = num_procs
  sfield_sorted = sfield.reshape(kprocs, jprocs, iprocs, zblocks, yblocks, xblocks)
  ## rearrange axes: [kprocs, jprocs, iprocs, zblocks, yblocks, xblocks] -> [kprocs*zblocks, jprocs*yblocks, iprocs*xblocks]
  sfield_sorted = sfield_sorted.swapaxes(1, 3).swapaxes(2, 4).reshape(
    kprocs * zblocks,
    jprocs * yblocks,
    iprocs * xblocks
  )
  return numpy.moveaxis(sfield_sorted, [0, 1, 2], [2, 1, 0])

def reformat_flash_sfield_v3(
    sfield     : numpy.ndarray,
    num_blocks : tuple[int, int, int],
    num_procs  : tuple[int, int, int],
  ):
  xblocks, yblocks, zblocks = num_blocks
  iprocs, jprocs, kprocs = num_procs
  sfield = sfield.transpose(0, 2, 1, 3)
  sfield_sorted = sfield.reshape(
    kprocs, jprocs, iprocs,
    zblocks, yblocks, xblocks
  )
  sfield_sorted = sfield_sorted.swapaxes(1, 3).swapaxes(2, 4).reshape(
    kprocs * zblocks,
    jprocs * yblocks,
    iprocs * xblocks
  )
  return numpy.transpose(sfield_sorted, [2, 1, 0])

def loadFlashDataCube(
    file_path    : str,
    field_name   : str,
    num_blocks   : tuple[int, int, int],
    num_procs    : tuple[int, int, int],
    print_h5keys : bool = False,
  ):
  ## open hdf5 file stream
  with h5py.File(file_path, "r") as h5file:
    ## create list of field-keys to extract from hdf5 file
    list_keys_stored = list(h5file.keys())
    list_keys_used = [
      key
      for key in list_keys_stored
      if key.startswith(field_name)
    ]
    if len(list_keys_used) == 0: raise Exception(f"Error: field-name '{field_name}' not found in {file_path}")
    ## check which keys are stored
    if print_h5keys: 
      print("--------- All the keys stored in the FLASH hdf5 file:\n\t" + "\n\t".join(list_keys_stored))
      print("--------- All the keys that were used: " + str(list_keys_used))
    ## extract fields from hdf5 file
    field_group_comp = [
      numpy.array(h5file[key])
      for key in list_keys_used
    ]
    ## close file stream
    h5file.close()
  ## reformat values
  field_sorted_group_comps = []
  for field_comp in field_group_comp:
    field_comp_sorted = reformat_flash_sfield_v1(field_comp, num_blocks, num_procs)
    field_sorted_group_comps.append(field_comp_sorted)
  ## return spatial-components of values
  return numpy.squeeze(field_sorted_group_comps)

# def loadVIData(
#     directory, t_turb,
#     field_index  = None,
#     field_name   = None,
#     time_start   = 1,
#     time_end     = numpy.inf,
#     bool_debug   = False,
#     verbose = False
#   ):
#   ## define which quantities to read in
#   time_index = 0
#   if field_index is None:
#     ## check that a variable name has been provided
#     if field_name is None: raise Exception("Error: need to provide either a field-index or field-name")
#     ## check which formatting the output file uses
#     with open(f"{directory}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
#       file_first_line = fp.readline()
#       bool_format_new = "#01_time" in file_first_line.split() # new version of file indexes from 1
#     ## get index of field in file
#     if   "kin"  in field_name.lower(): field_index = 9  if bool_format_new else 6
#     elif "mag"  in field_name.lower(): field_index = 11 if bool_format_new else 29
#     elif "mach" in field_name.lower(): field_index = 13 if bool_format_new else 8
#     else: raise Exception(f"Error: reading in {FileNames.FILENAME_FLASH_VI_DATA}")
#   ## initialise quantities to track traversal
#   data_time  = []
#   data_field = []
#   prev_time  = numpy.inf
#   with open(f"{directory}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
#     num_fields = len(fp.readline().split())
#     ## read values in backwards
#     for line in reversed(fp.readlines()):
#       data_split_columns = line.replace("\n", "").split()
#       ## only read where every field has been processed
#       if not(len(data_split_columns) == num_fields): continue
#       ## ignore comments
#       if "#" in data_split_columns[time_index]:  continue
#       if "#" in data_split_columns[field_index]: continue
#       ## compute time in units of eddy turnover time
#       this_time = float(data_split_columns[time_index]) / t_turb
#       ## only read values that has progressed in time
#       if this_time < prev_time:
#         data_val = float(data_split_columns[field_index])
#         ## something might have gone wrong: it is very unlikely to encounter a 0-value exactly
#         if (data_val == 0.0) and (0 < this_time):
#           warning_message = f"{FileNames.FILENAME_FLASH_VI_DATA}: value of field-index = {field_index} is 0.0 at time = {this_time}"
#           if bool_debug: raise Exception(f"Error: {warning_message}")
#           if verbose: print(f"Warning: {warning_message}")
#           continue
#         ## store values
#         data_time.append(this_time)
#         data_field.append(data_val)
#         ## step backwards
#         prev_time = this_time
#   ## re-order values
#   data_time  = data_time[::-1]
#   data_field = data_field[::-1]
#   ## subset values based on provided time bounds
#   index_start = list_utils.get_index_of_closest_value(data_time, time_start)
#   index_end   = list_utils.get_index_of_closest_value(data_time, time_end)
#   data_time_subset  = data_time[index_start  : index_end]
#   data_field_subset = data_field[index_start : index_end]
#   return data_time_subset, data_field_subset

# def loadSpectrum(file_path, spectrum_name, spectrum_component="total"):
#   with open(file_path, "r") as fp:
#     dataset = fp.readlines()
#     ## find row where header details are printed
#     header_index = next((
#       line_index+1
#       for line_index, line_contents in enumerate(list(dataset))
#       if "#" in line_contents
#     ), None)
#     if header_index is None: raise Exception("Error: no instances of `#` (which indicates the header was not) found in:", file_path)
#     ## read main dataset
#     values = numpy.array([
#       lines.strip().split() # remove leading/trailing whitespace + separate by whitespace-delimiter
#       for lines in dataset[header_index:] # read from after header
#     ])
#     ## get the indices assiated with fields of interest
#     iproc_index_ = 1
#     if   "lgt" in spectrum_component.lower(): field_index = 11 # longitudinal
#     elif "trv" in spectrum_component.lower(): field_index = 13 # transverse
#     elif "tot" in spectrum_component.lower():
#       if "SpectFunctTot".lower() in dataset[5].lower():
#         field_index = 15 # total = longitudinal + transverse
#       else: field_index = 7 # if there is no spectrum decomposition
#     else: raise Exception(f"Error: {spectrum_component} is an invalid spectra component.")
#     ## read fields from file
#     data_k     = numpy.array(values[:, iproc_index_], dtype=float)
#     data_power = numpy.array(values[:, field_index], dtype=float)
#     if   "vel" in spectrum_name.lower(): data_power = data_power / 2
#     elif "kin" in spectrum_name.lower(): data_power = data_power / 2
#     elif "mag" in spectrum_name.lower(): data_power = data_power / (8 * numpy.pi)
#     elif "cur" in spectrum_name.lower(): data_power = data_power / (4 * numpy.pi)
#     # elif "rho" in spectrum_name.lower(): data_power = data_power
#     else: raise Exception(f"Error: {spectrum_name} is an invalid spectra field. Failed to read and process:", file_path)
#     return data_k, data_power


## END OF LIBRARY