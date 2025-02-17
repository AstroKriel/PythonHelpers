## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, functools, fnmatch, h5py
import numpy as np
import xarray as xr
import multiprocessing as mproc
import concurrent.futures as cfut

## import user defined modules
from TheUsefulModule import WWFnF, WWFuncs, WWVariables, WWObjs
from TheFlashModule import LoadData, FileNames
from ThePlottingModule import PlotFuncs


## ###############################################################
## READ / UPDATE DRIVING PARAMETERS
## ###############################################################
def readDrivingAmplitude(directory):
  filepath_file = f"{directory}/{FileNames.FILENAME_DRIVING_INPUT}"
  ## check the file exists
  if not os.path.isfile(filepath_file):
    raise Exception("Error: turbulence driving input file does not exist:", FileNames.FILENAME_DRIVING_INPUT)
  ## open file
  with open(filepath_file) as fp:
    for line in fp.readlines():
      list_line_elems = line.split()
      ## ignore empty lines
      if len(list_line_elems) == 0: continue
      ## read driving amplitude
      if list_line_elems[0] == "ampl_factor": return float(list_line_elems[2])
  raise Exception(f"Error: could not read `ampl_factor` in the turbulence generator")

def updateDrivingAmplitude(directory, driving_amplitude):
  filepath_file = f"{directory}/{FileNames.FILENAME_DRIVING_INPUT}"
  ## read previous driving parameters
  list_lines = []
  with open(filepath_file, "r") as fp:
    for line in fp.readlines():
      if "ampl_factor" in line:
        new_line = line.replace(line.split("=")[1].split()[0], str(driving_amplitude))
        list_lines.append(new_line)
      else: list_lines.append(line)
  ## write updated driving paramaters
  with open(filepath_file, "w") as output_file:
    output_file.writelines(list_lines)

def updateDrivingHistory(directory, current_time, measured_Mach, old_driving_amplitude, new_driving_amplitude):
  filepath_file = f"{directory}/{FileNames.FILENAME_DRIVING_HISTORY}"
  with open(filepath_file, "a") as fp:
    fp.write(f"{current_time} {measured_Mach} {old_driving_amplitude} {new_driving_amplitude}\n")


## ###############################################################
## GET FILE NUMBERS FROM FLASH OUTPUTS
## ###############################################################
def getLastChkFilename(directory):
  list_chkfilenames = [
    filename
    for filename in os.listdir(directory)
    if fnmatch.fnmatch(filename, "*_hdf5_chk_*")
  ]
  if len(list_chkfilenames) == 0: raise Exception("Error: No checkpoint file found.")
  list_chkfilenames.sort()
  return list_chkfilenames[len(list_chkfilenames)-1]


## ###############################################################
## CREATE STRINGS FROM SIMULATION PARAMETERS
## ###############################################################
def getSonicRegime(Mach):
  if Mach < 1: return f"Mach{float(Mach):.1f}"
  return f"Mach{int(Mach):d}"

def getJobTag(dict_sim_inputs, job_name):
  mach_folder  = getSonicRegime(dict_sim_inputs["desired_Mach"])
  suite_folder = dict_sim_inputs["suite_folder"]
  sim_folder   = dict_sim_inputs["sim_folder"]
  res_folder   = dict_sim_inputs["res_folder"]
  return f"{mach_folder}{suite_folder}{sim_folder}Nres{res_folder}{job_name}"

def getSimName(dict_sim_inputs, bool_include_res=True):
  return "{}{}{}{}".format(
    dict_sim_inputs["mach_folder"],
    dict_sim_inputs["suite_folder"],
    dict_sim_inputs["sim_folder"],
    "Nres" + dict_sim_inputs["res_folder"] if bool_include_res else "",
  )


## ###############################################################
## CREATE A LIST OF SIMULATION DIRECTORIES
## ###############################################################
def getListOfSimDirectories(list_base_paths, list_suite_folders, list_mach_folders, list_sim_folders, list_res_folders):
  return [
    WWFnF.createFilepath([ base_path, suite_folder, mach_folder, sim_folder, res_folder ])
    for base_path    in list_base_paths
    for suite_folder in list_suite_folders
    for mach_folder  in list_mach_folders
    for sim_folder   in list_sim_folders
    for res_folder   in list_res_folders
    if os.path.exists(
      WWFnF.createFilepath([ base_path, suite_folder, mach_folder, sim_folder, res_folder ])
    )
  ]


## ###############################################################
## APPLY FUNCTION OVER ALL SIMULATIONS
## ###############################################################
def callFuncForAllDirectories(func, list_directories, bool_debug_mode=False):
  with cfut.ProcessPoolExecutor() as executor:
    manager = mproc.Manager()
    lock = manager.Lock()
    list_jobs = [
      executor.submit(
        functools.partial(
          func,
          lock            = lock,
          bool_debug_mode = bool_debug_mode,
          bool_verbose    = False
        ),
        directory
      ) for directory in list_directories
    ]
    ## wait to ensure that all scheduled and running tasks have completed
    cfut.wait(list_jobs)
    ## check if any tasks failed
    for job in cfut.as_completed(list_jobs):
      try:
        job.result()
      except Exception as e: print(f"Job failed with exception: {e}")

def callFuncForAllSimulations(
    func,
    list_base_paths, list_suite_folders, list_mach_folders, list_sim_folders, list_res_folders,
    bool_mproc      = False,
    bool_debug_mode = False
  ):
  list_directory_sims = getListOfSimDirectories(
    list_base_paths    = list_base_paths,
    list_suite_folders = list_suite_folders,
    list_mach_folders  = list_mach_folders,
    list_sim_folders   = list_sim_folders,
    list_res_folders   = list_res_folders
  )
  if bool_mproc:
    print(f"Looking at {len(list_directory_sims)} simulation(s):")
    [
      print("\t> " + directory_sim)
      for directory_sim in list_directory_sims
    ]
    print(" ")
    print("Processing...")
    callFuncForAllDirectories(
      func             = func,
      list_directories = list_directory_sims,
      bool_debug_mode  = bool_debug_mode
    )
    print("Finished processing.")
  else: [
    func(
      directory_sim    = directory_sim,
      bool_debug_mode  = bool_debug_mode,
      bool_verbose     = True
    )
    for directory_sim in list_directory_sims
  ]


## ###############################################################
## PLOT SIMULATION DETAILS
## ###############################################################
def addLabel_simInputs(
    fig, ax,
    dict_sim_inputs = None,
    directory        = None,
    bbox            = (0,0),
    vpos            = (0.05, 0.05),
    bool_show_res   = True
  ):
  ## load simulation parameters
  if dict_sim_inputs is None:
    if directory is None:
      raise Exception("Error: need to pass details about simulation inputs")
    dict_sim_inputs = readSimInputs(directory)
  ## annotate simulation parameters
  Nres = int(dict_sim_inputs["res_folder"].split("v")[0])
  PlotFuncs.addBoxOfLabels(
    fig, ax,
    bbox        = bbox,
    xpos        = vpos[0],
    ypos        = vpos[1],
    alpha       = 0.5,
    fontsize    = 18,
    list_labels = [
      r"${\rm N}_{\rm res} = $ " + "{:d}".format(Nres) if bool_show_res else "",
      r"${\rm Re} = $ " + "{:d}".format(int(dict_sim_inputs["Re"])),
      r"${\rm Rm} = $ " + "{:d}".format(int(dict_sim_inputs["Rm"])),
      r"${\rm Pm} = $ " + "{:d}".format(int(dict_sim_inputs["Pm"])),
      r"$\mathcal{M} = $ " + "{:.1f}".format(dict_sim_inputs["desired_Mach"]),
    ]
  )


## ###############################################################
## SAVE + READ SIMULALATION FILES
## ###############################################################
# def addSpectrum2Xarray(ds, dict_spectrum, spectrum_name, bool_overwrite=False):
#   array_t_turb        = np.array(dict_spectrum["list_t_turb"])
#   array_k_turb        = np.array(dict_spectrum["list_k_turb"])
#   array_power_group_t = np.array(dict_spectrum["spectra_group_t"])
#   if len(array_t_turb) != len(array_power_group_t):
#     raise ValueError("Error: Mismatched lengths of `list_t_turb` and `spectra_group_t`.")
#   new_data_array = xr.DataArray(
#     data   = array_power_group_t,
#     dims   = [ "array_t_turb", "array_k_turb" ],
#     coords = {
#       "array_t_turb" : array_t_turb,
#       "array_k_turb" : array_k_turb,
#     },
#   )
#   if spectrum_name in ds:
#     if bool_overwrite: ds[spectrum_name] = new_data_array
#     else:
#       ## merge time points
#       _array_t_turb = np.union1d(ds["array_t_turb"].values, array_t_turb)
#       ## reindex both the existing and new data arrays to accommodate for new time points
#       old_data_array = ds[spectrum_name].reindex(array_t_turb=_array_t_turb, method=None)
#       new_data_array = new_data_array.reindex(array_t_turb=_array_t_turb, method=None)
#       ## combine the data and have new data take precedence
#       merged_data_array = new_data_array.combine_first(old_data_array)
#       ## update the data array
#       ds = ds.reindex(array_t_turb=_array_t_turb, method=None)
#       ds[spectrum_name] = merged_data_array
#   else:
#     ## ensure time alignment and add it
#     if "array_t_turb" in ds.coords:
#       _array_t_turb = np.union1d(ds["array_t_turb"].values, array_t_turb)
#     else: _array_t_turb = array_t_turb
#     ds.reindex(array_t_turb=_array_t_turb, method=None)
#     ds[spectrum_name] = new_data_array

@WWFuncs.warn_if_result_unused
def addSpectrum2Xarray(ds, dict_spectrum, spectrum_name, bool_overwrite=False):
  array_t_turb = np.array(dict_spectrum["list_t_turb"])
  array_k_turb = np.array(dict_spectrum["list_k_turb"])
  array_power_group_t = np.array(dict_spectrum["spectra_group_t"])
  if len(array_t_turb) != len(array_power_group_t):
    raise ValueError("Error: Mismatched lengths of `list_t_turb` and `spectra_group_t`.")
  new_data_array = xr.DataArray(
    data=array_power_group_t,
    dims=["array_t_turb", "array_k_turb"],
    coords={
      "array_t_turb": array_t_turb,
      "array_k_turb": array_k_turb,
    },
  )
  if len(ds.data_vars) == 0:
    ds[spectrum_name] = new_data_array
    return ds
  if "array_t_turb" in ds.coords:
    _array_t_turb = np.union1d(ds["array_t_turb"].values, array_t_turb)
    ds = ds.reindex(array_t_turb=_array_t_turb, method=None)
    new_data_array = new_data_array.reindex(array_t_turb=_array_t_turb, method=None)
  else:
    ds = ds.assign_coords(array_t_turb=array_t_turb)
  if spectrum_name not in ds:
    ds[spectrum_name] = new_data_array
  else:
    if bool_overwrite:
      ds[spectrum_name] = new_data_array
    else:
      existing_data = ds[spectrum_name]
      merged_data = new_data_array.combine_first(existing_data)
      ds[spectrum_name] = merged_data
  return ds

def saveSimOutputs(ds, directory, bool_verbose=True):
  filepath_file = f"{directory}/{FileNames.FILENAME_SIM_OUTPUTS}"
  with h5py.File(filepath_file, "w") as hdf_file:
    ## save dataset variables
    for var_name, var_data in ds.data_vars.items():
      hdf_file.create_dataset(var_name, data=var_data.values)
      dims_str = ",".join(var_data.dims)
      hdf_file[var_name].attrs["dims"] = dims_str
    ## save table-coordinates
    for coord_name, coord_data in ds.coords.items():
      hdf_file.create_dataset(f"coords/{coord_name}", data=coord_data.values)
    ## save global attributes of the dataset
    for attr_name, attr_value in ds.attrs.items():
      hdf_file.attrs[attr_name] = attr_value
  if bool_verbose: print("Saved dataset:", filepath_file)

def readSimOutputs(directory, bool_verbose=True):
  filepath_file = f"{directory}/{FileNames.FILENAME_SIM_OUTPUTS}"
  if bool_verbose: print("Reading:", filepath_file)
  data_vars = {}
  coords = {}
  with h5py.File(filepath_file, "r") as hdf_file:
    ## explicityly read the table-coordinates (e.g., t_turb, k_turb)
    if "coords/array_t_turb" in hdf_file: coords["array_t_turb"] = hdf_file["coords/array_t_turb"][:]
    if "coords/array_k_turb" in hdf_file: coords["array_k_turb"] = hdf_file["coords/array_k_turb"][:]
    ## read data variables
    for var_name in hdf_file.keys():
      ## skip groups (like "coords/") and only read datasets
      if isinstance(hdf_file[var_name], h5py.Group): continue
      data = hdf_file[var_name][:]
      dims_str = hdf_file[var_name].attrs.get("dims", "")
      dims = dims_str.split(",")
      data_vars[var_name] = (dims, data)
  ## manually create a Xarray-Dataset + assign the correct coordinates
  ds = xr.Dataset(
    data_vars = data_vars,
    coords    = coords
  )
  return ds

def saveSimSummary(directory, dict_sim_summary):
  filepath_file = f"{directory}/{FileNames.FILENAME_SIM_SUMMARY}"
  WWObjs.saveDict2JsonFile(filepath_file, dict_sim_summary)

def readSimSummary(directory, bool_verbose=True):
  dict_sim_summary = WWObjs.readJsonFile2Dict(
    directory     = directory,
    filename     = FileNames.FILENAME_SIM_SUMMARY,
    bool_verbose = bool_verbose
  )
  return dict_sim_summary

def saveSimInputs(directory, sim_inputs):
  filepath_file = f"{directory}/{FileNames.FILENAME_SIM_INPUTS}"
  if   type(sim_inputs) is dict:           WWObjs.saveDict2JsonFile(filepath_file, sim_inputs)
  elif type(sim_inputs) is SimInputParams: WWObjs.saveObj2JsonFile(filepath_file, sim_inputs)

def readSimInputs(directory, bool_verbose=True):
  dict_sim_inputs = WWObjs.readJsonFile2Dict(
    directory     = directory,
    filename     = FileNames.FILENAME_SIM_INPUTS,
    bool_verbose = bool_verbose
  )
  ## make sure that every parameter is in the inputs class is defined
  obj_sim_inputs = SimInputParams(**dict_sim_inputs)
  ## save updated dictionary if new parameters were defined
  bool_new_params_defined = WWObjs.areDictsDifferent(dict_sim_inputs, obj_sim_inputs.__dict__)
  if bool_new_params_defined: saveSimInputs(directory, obj_sim_inputs)
  ## return dictionary object
  return obj_sim_inputs.__dict__

def createSimInputs(
    directory, suite_folder, sim_folder, res_folder, desired_Mach, k_turb,
    Re = None,
    Rm = None,
    Pm = None
  ):
  ## check that a valid driving scale is defined
  if k_turb is None: raise Exception(f"Error: you have provided a invalid driving scale:", k_turb)
  ## number of cells per block that the flash4-exe was compiled with
  _res_folder = res_folder.split("v")[0]
  if   _res_folder in [ "576", "1152" ]: num_blocks = [ 96, 96, 72 ]
  elif _res_folder in [ "144", "288" ]:  num_blocks = [ 36, 36, 48 ]
  elif _res_folder in [ "36", "72" ]:    num_blocks = [ 12, 12, 18 ]
  elif _res_folder in [ "18" ]:          num_blocks = [ 6, 6, 6 ]
  num_procs = [
    int(int(_res_folder) / num_blocks_in_dir)
    for num_blocks_in_dir in num_blocks
  ]
  ## create object to define simulation input parameters
  obj_sim_params = SimInputParams(
    suite_folder = suite_folder,
    sim_folder   = sim_folder,
    res_folder   = res_folder,
    desired_Mach = desired_Mach,
    k_turb       = k_turb,
    num_blocks   = num_blocks,
    num_procs    = num_procs,
    Re           = Re,
    Rm           = Rm,
    Pm           = Pm
  )
  obj_sim_params.defineParams()
  ## save input file
  saveSimInputs(directory, obj_sim_params)
  return obj_sim_params.__dict__


## ###############################################################
## COMPUTE ALL RELEVANT SIMULATION PARAMETERS
## ###############################################################
class SimInputParams():
  def __init__(
      self,
      suite_folder, sim_folder, res_folder, desired_Mach, k_turb, num_blocks, num_procs,
      bool_driving_tuned = None,
      run_index          = None,
      cfl                = None,
      max_num_t_turb     = None,
      outputs_per_t_turb = None,
      init_rms_b         = None,
      mach_folder        = None,
      t_turb             = None,
      Re                 = None,
      Rm                 = None,
      Pm                 = None,
      nu                 = None,
      eta                = None,
      **kwargs # unused arguments
    ):
    def defineNoneParam(param_name, default_value):
      return param_name if (param_name is not None) else default_value
    ## required parameters
    self.suite_folder = suite_folder
    self.sim_folder   = sim_folder
    self.res_folder   = res_folder
    self.desired_Mach = desired_Mach
    self.k_turb       = k_turb
    self.num_blocks   = num_blocks
    self.num_procs    = num_procs
    ## optional parameters (with default values)
    self.bool_driving_tuned = defineNoneParam(bool_driving_tuned, False)
    self.run_index          = defineNoneParam(run_index, 0)
    self.cfl                = defineNoneParam(cfl, 0.8)
    self.max_num_t_turb     = defineNoneParam(max_num_t_turb, 100)
    self.outputs_per_t_turb = defineNoneParam(outputs_per_t_turb, 10)
    ## parameters that (may) need to be computed
    self.init_rms_b  = init_rms_b
    self.mach_folder = mach_folder
    self.t_turb      = t_turb
    self.Re          = Re
    self.Rm          = Rm
    self.Pm          = Pm
    self.nu          = nu
    self.eta         = eta

  def defineParams(self):
    ## check that the required input arguments are the right type
    WWVariables.assertType("suite_folder", self.suite_folder, str)
    WWVariables.assertType("sim_folder",   self.sim_folder,   str)
    WWVariables.assertType("res_folder",   self.res_folder,   str)
    WWVariables.assertType("num_blocks",   self.num_blocks,   list)
    WWVariables.assertType("k_turb",       self.k_turb,       (int, float))
    WWVariables.assertType("Mach",         self.desired_Mach, (int, float))
    ## perform routines
    list_undefined_plasma_params = [
      param is None
      for param in [
        self.Re, self.Rm, self.Pm, self.nu, self.eta
      ]
    ]
    if self.init_rms_b is None: self.__defineInitBEnergy()
    if (self.t_turb is None) or (self.mach_folder is None): self.__defineSonicRegime()
    if any(list_undefined_plasma_params): self.__definePlasmaParameters()
    self.__checkSimParamsDefined()
    self.__roundParams()

  def __defineInitBEnergy(self):
    if self.desired_Mach < 1: self.init_rms_b = 1e-7
    else:                     self.init_rms_b = 1e-5

  def __defineSonicRegime(self):
    self.mach_folder = getSonicRegime(self.desired_Mach)
    self.t_turb = 1 / (self.k_turb * self.desired_Mach) # c_s = 1

  def __definePlasmaParameters(self):
    dict_params = LoadData.computePlasmaConstants(
      Mach   = self.desired_Mach,
      k_turb = self.k_turb,
      Re     = self.Re,
      Rm     = self.Rm,
      Pm     = self.Pm
    )
    self.Re  = dict_params["Re"]
    self.Rm  = dict_params["Rm"]
    self.Pm  = dict_params["Pm"]
    self.nu  = dict_params["nu"]
    self.eta = dict_params["eta"]

  def __checkSimParamsDefined(self):
    list_check_params_defined = [
      "init_rms_b"   if self.init_rms_b   is None else "",
      "mach_folder"  if self.mach_folder  is None else "",
      "t_turb"       if self.t_turb       is None else "",
      "Re"           if self.Re           is None else "",
      "Rm"           if self.Rm           is None else "",
      "Pm"           if self.Pm           is None else "",
      "nu"           if self.nu           is None else "",
      "eta"          if self.eta          is None else ""
    ]
    list_params_not_defined = [
      param_name
      for param_name in list_check_params_defined
      ## remove entry if it is empty
      if len(param_name) > 0
    ]
    if len(list_params_not_defined) > 0:
      raise Exception(f"Error: You have not defined the following parameter(s):", list_params_not_defined)

  def __roundParams(self, num_decimals=5):
    ## round numeric parameter values
    self.k_turb       = round(self.k_turb,       num_decimals)
    self.desired_Mach = round(self.desired_Mach, num_decimals)
    self.t_turb       = round(self.t_turb,       num_decimals)
    self.Re           = round(self.Re,           num_decimals)
    self.Rm           = round(self.Rm,           num_decimals)
    self.Pm           = round(self.Pm,           num_decimals)
    self.nu           = round(self.nu,           num_decimals)
    self.eta          = round(self.eta,          num_decimals)


## END OF LIBRARY