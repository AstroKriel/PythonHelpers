## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import h5py
import numpy


## ###############################################################
## FUNCTIONS FOR LOADING FLASH DATA
## ###############################################################
def readFromChkFile(fielpath_chkfile, dset, param):
  with h5py.File(fielpath_chkfile, "r") as h5file:
    for _param, _value in h5file[dset]:
      if f"b'{param}" in str(_param): return int(_value)
  return None

def reformatFlashField(field, num_blocks, num_procs):
  xblocks, yblocks, zblocks = num_blocks
  iprocs, jprocs, kprocs = num_procs
  ## initialise the organised field
  field_sorted = numpy.zeros(
    shape = (
      zblocks*kprocs,
      yblocks*jprocs,
      xblocks*iprocs,
    ),
    dtype = numpy.float32,
    order = "C"
  )
  ## [ num_cells_per_block, yblocks, zblocks, xblocks ] -> [ kprocs*zblocks, jprocs*yblocks, iprocs*xblocks ]
  block_index = 0
  for kproc_index in range(kprocs):
    for jproc_index in range(jprocs):
      for iproc_index in range(iprocs):
        field_sorted[
          kproc_index * zblocks : (kproc_index+1) * zblocks,
          jproc_index * yblocks : (jproc_index+1) * yblocks,
          iproc_index * xblocks : (iproc_index+1) * xblocks,
        ] = field[block_index, :, :, :]
        block_index += 1
  ## rearrange indices [ z, y, x ] -> [ x, y, z ]
  field_sorted = numpy.transpose(field_sorted, [2, 1, 0])
  return field_sorted

@WWFuncs.time_function
def loadFlashDataCube(
    filepath_file, num_blocks, num_procs, field_name,
    bool_norm_rms     = False,
    bool_print_h5keys = False
  ):
  ## open hdf5 file stream
  with h5py.File(filepath_file, "r") as h5file:
    ## create list of field-keys to extract from hdf5 file
    list_keys_stored = list(h5file.keys())
    list_keys_used = [
      key
      for key in list_keys_stored
      if key.startswith(field_name)
    ]
    if len(list_keys_used) == 0: raise Exception(f"Error: field-name '{field_name}' not found in {filepath_file}")
    ## check which keys are stored
    if bool_print_h5keys: 
      print("--------- All the keys stored in the FLASH hdf5 file:\n\t" + "\n\t".join(list_keys_stored))
      print("--------- All the keys that were used: " + str(list_keys_used))
    ## extract fields from hdf5 file
    field_group_comp = [
      numpy.array(h5file[key])
      for key in list_keys_used
    ]
    ## close file stream
    h5file.close()
  ## reformat data
  field_sorted_group_comps = []
  for field_comp in field_group_comp:
    field_comp_sorted = reformatFlashField(field_comp, num_blocks, num_procs)
    ## normalise by rms-value
    if bool_norm_rms: field_comp_sorted /= WWFields.sfieldRMS(field_comp_sorted)
    field_sorted_group_comps.append(field_comp_sorted)
  ## return spatial-components of data
  return numpy.squeeze(field_sorted_group_comps)

def loadAllFlashDataCubes(
    directory, field_name, dict_sim_config,
    start_time = 0,
    end_time   = numpy.inf,
    read_every = 1
  ):
  outputs_per_t_turb = dict_sim_config["outputs_per_t_turb"]
  ## get all plt files in the directory
  list_filenames = WWFnF.getFilesInDirectory(
    directory             = directory,
    filename_starts_with  = FileNames.FILENAME_FLASH_PLT_FILES,
    filename_not_contains = "spect",
    loc_file_index        = 4,
    file_start_index      = outputs_per_t_turb * start_time,
    file_end_index        = outputs_per_t_turb * end_time
  )
  ## find min and max colorbar limits
  field_min = numpy.nan
  field_max = numpy.nan
  ## save field slices and simulation times
  field_group_t = []
  list_t_turb   = []
  for filename in list_filenames[::read_every]:
    field_magnitude = loadFlashDataCube(
      filepath_file = f"{directory}/{filename}",
      num_blocks    = dict_sim_config["num_blocks"],
      num_procs     = dict_sim_config["num_procs"],
      field_name    = field_name
    )
    list_t_turb.append( float(filename.split("_")[-1]) / outputs_per_t_turb )
    field_group_t.append( field_magnitude[:,:] )
    field_min = numpy.nanmin([ field_min, numpy.nanmin(field_magnitude[:,:]) ])
    field_max = numpy.nanmax([ field_max, numpy.nanmax(field_magnitude[:,:]) ])
  print(" ")
  return {
    "list_t_turb" : list_t_turb,
    "field_group_t"   : field_group_t,
    "field_bounds"    : [ field_min, field_max ]
  }

def loadVIData(
    directory, t_turb,
    field_index  = None,
    field_name   = None,
    time_start   = 1,
    time_end     = numpy.inf,
    bool_debug   = False,
    bool_verbose = False
  ):
  ## define which quantities to read in
  time_index = 0
  if field_index is None:
    ## check that a variable name has been provided
    if field_name is None: raise Exception("Error: need to provide either a field-index or field-name")
    ## check which formatting the output file uses
    with open(f"{directory}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
      file_first_line = fp.readline()
      bool_format_new = "#01_time" in file_first_line.split() # new version of file indexes from 1
    ## get index of field in file
    if   "kin"  in field_name.lower(): field_index = 9  if bool_format_new else 6
    elif "mag"  in field_name.lower(): field_index = 11 if bool_format_new else 29
    elif "mach" in field_name.lower(): field_index = 13 if bool_format_new else 8
    else: raise Exception(f"Error: reading in {FileNames.FILENAME_FLASH_VI_DATA}")
  ## initialise quantities to track traversal
  data_time  = []
  data_field = []
  prev_time  = numpy.inf
  with open(f"{directory}/{FileNames.FILENAME_FLASH_VI_DATA}", "r") as fp:
    num_fields = len(fp.readline().split())
    ## read data in backwards
    for line in reversed(fp.readlines()):
      data_split_columns = line.replace("\n", "").split()
      ## only read where every field has been processed
      if not(len(data_split_columns) == num_fields): continue
      ## ignore comments
      if "#" in data_split_columns[time_index]:  continue
      if "#" in data_split_columns[field_index]: continue
      ## compute time in units of eddy turnover time
      this_time = float(data_split_columns[time_index]) / t_turb
      ## only read data that has progressed in time
      if this_time < prev_time:
        data_val = float(data_split_columns[field_index])
        ## something might have gone wrong: it is very unlikely to encounter a 0-value exactly
        if (data_val == 0.0) and (0 < this_time):
          warning_message = f"{FileNames.FILENAME_FLASH_VI_DATA}: value of field-index = {field_index} is 0.0 at time = {this_time}"
          if bool_debug: raise Exception(f"Error: {warning_message}")
          if bool_verbose: print(f"Warning: {warning_message}")
          continue
        ## store data
        data_time.append(this_time)
        data_field.append(data_val)
        ## step backwards
        prev_time = this_time
  ## re-order data
  data_time  = data_time[::-1]
  data_field = data_field[::-1]
  ## subset data based on provided time bounds
  index_start = WWLists.getIndexOfClosestValue(data_time, time_start)
  index_end   = WWLists.getIndexOfClosestValue(data_time, time_end)
  data_time_subset  = data_time[index_start  : index_end]
  data_field_subset = data_field[index_start : index_end]
  return data_time_subset, data_field_subset

def loadSpectrum(filepath_file, spectrum_name, spectrum_component="total"):
  with open(filepath_file, "r") as fp:
    dataset = fp.readlines()
    ## find row where header details are printed
    header_index = next((
      line_index+1
      for line_index, line_contents in enumerate(list(dataset))
      if "#" in line_contents
    ), None)
    if header_index is None: raise Exception("Error: no instances of `#` (which indicates the header was not) found in:", filepath_file)
    ## read main dataset
    data = numpy.array([
      lines.strip().split() # remove leading/trailing whitespace + separate by whitespace-delimiter
      for lines in dataset[header_index:] # read from after header
    ])
    ## get the indices assiated with fields of interest
    iproc_index_ = 1
    if   "lgt" in spectrum_component.lower(): field_index = 11 # longitudinal
    elif "trv" in spectrum_component.lower(): field_index = 13 # transverse
    elif "tot" in spectrum_component.lower():
      if "SpectFunctTot".lower() in dataset[5].lower():
        field_index = 15 # total = longitudinal + transverse
      else: field_index = 7 # if there is no spectrum decomposition
    else: raise Exception(f"Error: {spectrum_component} is an invalid spectra component.")
    ## read fields from file
    data_k     = numpy.array(data[:, iproc_index_], dtype=float)
    data_power = numpy.array(data[:, field_index], dtype=float)
    if   "vel" in spectrum_name.lower(): data_power = data_power / 2
    elif "kin" in spectrum_name.lower(): data_power = data_power / 2
    elif "mag" in spectrum_name.lower(): data_power = data_power / (8 * numpy.pi)
    elif "cur" in spectrum_name.lower(): data_power = data_power / (4 * numpy.pi)
    # elif "rho" in spectrum_name.lower(): data_power = data_power
    else: raise Exception(f"Error: {spectrum_name} is an invalid spectra field. Failed to read and process:", filepath_file)
    return data_k, data_power

def loadAllSpectra(
    directory, spectrum_name, outputs_per_t_turb,
    spectrum_component = "total",
    file_start_time    = 2,
    file_end_time      = numpy.inf,
    read_every         = 1,
    bool_verbose       = True
  ):
  if   "vel" in spectrum_name.lower(): file_end_str = "spect_velocity.dat"
  elif "kin" in spectrum_name.lower(): file_end_str = "spect_kinetic.dat"
  elif "mag" in spectrum_name.lower(): file_end_str = "spect_magnetic.dat"
  elif "cur" in spectrum_name.lower(): file_end_str = "spect_current.dat"
  # elif "rho" in spectrum_name.lower(): file_end_str = "spect_density.dat"
  else: raise Exception("Error: invalid spectra field-type provided:", spectrum_name)
  ## get list of spect-filenames in directory
  list_spectra_filenames = WWFnF.getFilesInDirectory(
    directory          = directory,
    filename_ends_with = file_end_str,
    loc_file_index     = -3,
    file_start_index   = outputs_per_t_turb * file_start_time,
    file_end_index     = outputs_per_t_turb * file_end_time
  )
  ## initialise list of spectra data
  list_t_turb        = []
  list_k_turb        = None
  spectra_group_t = []
  ## loop over each of the spectra file names
  for filename in list_spectra_filenames[::read_every]:
    ## convert file index to simulation time
    turb_time = float(filename.split("_")[-3]) / outputs_per_t_turb
    ## load data
    list_k_turb, spectrum = loadSpectrum(
      filepath_file = f"{directory}/{filename}",
      spectrum_name   = spectrum_name,
      spectrum_component    = spectrum_component
    )
    ## store data
    spectra_group_t.append(spectrum)
    list_t_turb.append(turb_time)
  ## return spectra data
  return {
    "list_t_turb"        : list_t_turb,
    "list_k_turb"        : list_k_turb,
    "spectra_group_t" : spectra_group_t,
  }

def computePlasmaConstants(Mach, k_turb, Re=None, Rm=None, Pm=None):
  ## Re and Pm have been defined
  if (Re is not None) and (Pm is not None):
    Re  = float(Re)
    Pm  = float(Pm)
    Rm  = Re * Pm
    nu  = round(Mach / (k_turb * Re), 5)
    eta = round(nu / Pm, 5)
  ## Rm and Pm have been defined
  elif (Rm is not None) and (Pm is not None):
    Rm  = float(Rm)
    Pm  = float(Pm)
    Re  = Rm / Pm
    eta = round(Mach / (k_turb * Rm), 5)
    nu  = round(eta * Pm, 5)
  ## error
  else: raise Exception(f"Error: insufficient plasma Reynolds numbers provided: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
  return {
    "nu"  : nu,
    "eta" : eta,
    "Re"  : Re,
    "Rm"  : Rm,
    "Pm"  : Pm
  }

def computePlasmaNumbers(Re=None, Rm=None, Pm=None):
  if   (Re is not None) and (Pm is not None): Rm = Re * Pm
  elif (Rm is not None) and (Pm is not None): Re = Rm / Pm
  elif (Re is not None) and (Rm is not None): Pm = Rm / Re
  else: raise Exception(f"Error: insufficient plasma Reynolds numbers provided: Re = {Re}, Rm = {Rm}, Pm = {Rm}")
  return {
    "Re"  : Re,
    "Rm"  : Rm,
    "Pm"  : Pm
  }

def getNumberFromString(input_string, var_name):
  _input_string = input_string.lower()
  _var_name     = var_name.lower()
  if _var_name in _input_string:
    return float(_input_string.replace(_var_name, ""))
  else: return None



## END OF LIBRARY