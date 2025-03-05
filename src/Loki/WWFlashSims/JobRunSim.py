## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os

from Loki.TheFlashModule import ReadFlashData, FileNames, LoadData


## ###############################################################
## HELPFUL CONSTANTS
## ###############################################################
## turbulence driving input file
DRIVING_INPUT_NSPACES_PRE_ASSIGN  = 18
DRIVING_INPUT_NSPACES_PRE_COMMENT = 28
## flash simulation input file
FLASH_INPUT_NSPACES_PRE_ASSIGN    = 30
FLASH_INPUT_NSPACES_PRE_COMMENT   = 45


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def paramAssignLine(
    param_name, param_value,
    comment             = "",
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  space_assign  = " " * (nspaces_pre_assign - len(param_name))
  param_assign  = f"{param_name}{space_assign}= {param_value}"
  if len(comment) > 0:
    space_comment = " " * (nspaces_pre_comment - len(param_assign))
    return f"{param_assign}{space_comment}# {comment}\n"
  else: return f"{param_assign}\n"

def addParamAssign(
    dict_assigns, param_name, param_value,
    comment             = "",
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  assign_line = paramAssignLine(param_name, param_value, comment, nspaces_pre_assign, nspaces_pre_comment)
  dict_assigns[param_name] = {
    "assign_line"   : assign_line,
    "bool_assigned" : False
  }

def processLine(
    ref_line, dict_assigns,
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  list_line_elems = ref_line.split()
  ## empty line
  if len(list_line_elems) == 0:
    return "\n"
  ## parameter who's value needs to be assigned
  elif list_line_elems[0] in dict_assigns:
    param_name = list_line_elems[0]
    dict_assigns[param_name]["bool_assigned"] = True
    return str(dict_assigns[param_name]["assign_line"])
  ## parameter who's value doesn't need to change
  elif not(list_line_elems[0] == "#") and (list_line_elems[1] == "="):
    ## line format: param_name = param_value # comment
    param_name  = list_line_elems[0]
    param_value = list_line_elems[2]
    if (len(list_line_elems) > 4) and (list_line_elems[3] == "#"):
      comment = " ".join(list_line_elems[4:])
    else: comment = ""
    return paramAssignLine(param_name, param_value, comment, nspaces_pre_assign, nspaces_pre_comment)
  ## comment that has overflowed to the next line
  ## (* are used as headings)
  elif (list_line_elems[0] == "#") and ("*" not in ref_line):
    space_pre_comment = " " * nspaces_pre_comment
    comment = " ".join(list_line_elems[1:])
    return f"{space_pre_comment}# {comment}\n"
  ## something else (e.g., heading)
  return ref_line


## ###############################################################
## WRITE TURBULENCE DRIVING FILE
## ###############################################################
def writeTurbDrivingFile(filepath_ref, filepath_to, dict_params):
  ## helper function
  def _addParamAssign(param_name, param_value, comment):
    addParamAssign(
      dict_assigns        = dict_assigns,
      param_name          = param_name,
      param_value         = param_value,
      comment             = comment,
      nspaces_pre_assign  = DRIVING_INPUT_NSPACES_PRE_ASSIGN,
      nspaces_pre_comment = DRIVING_INPUT_NSPACES_PRE_COMMENT
    )
  ## initialise dictionary storing parameter assignemnts
  dict_assigns = {}
  ## add parameter assignments
  _addParamAssign(
    param_name  = "velocity",
    param_value = "{:.3f}".format(dict_params["des_velocity"]),
    comment     = "Target turbulent velocity dispersion"
  )
  _addParamAssign(
    param_name  = "ampl_factor",
    param_value = "{:.5f}".format(dict_params["des_ampl_factor"]),
    comment     = "Used to achieve a target velocity dispersion; scales with velocity/velocity_measured"
  )
  _addParamAssign(
    param_name  = "k_driv",
    param_value = "{:.3f}".format(dict_params["des_k_driv"]),
    comment     = "Characteristic driving scale in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "k_min",
    param_value = "{:.3f}".format(dict_params["des_k_min"]),
    comment     = "Minimum driving wavnumber in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "k_max",
    param_value = "{:.3f}".format(dict_params["des_k_max"]),
    comment     = "Maximum driving wavnumber in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "sol_weight",
    param_value = "{:.3f}".format(dict_params["des_sol_weight"]),
    comment     = "1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture"
  )
  _addParamAssign(
    param_name  = "spect_form",
    param_value = "{:.3f}".format(dict_params["des_spect_form"]),
    comment     = "0: band/rectangle/constant, 1: paraboloid, 2: power law"
  )
  _addParamAssign(
    param_name  = "nsteps_per_t_turb",
    param_value = "{:d}".format(dict_params["des_nsteps_per_t_turb"]),
    comment     = "Number of turbulence driving pattern updates per turnover time"
  )
  ## read from reference file and write to new files
  filepath_ref  = f"{filepath_ref}/{FileNames.FILENAME_DRIVING_INPUT}"
  filepath_file = f"{filepath_to}/{FileNames.FILENAME_DRIVING_INPUT}"
  with open(filepath_ref, "r") as ref_file:
    with open(filepath_file, "w") as new_file:
      for ref_line in ref_file.readlines():
        new_line = processLine(
          ref_line, dict_assigns,
          nspaces_pre_assign  = DRIVING_INPUT_NSPACES_PRE_ASSIGN,
          nspaces_pre_comment = DRIVING_INPUT_NSPACES_PRE_COMMENT
        )
        new_file.write(new_line)
  ## check that every parameter has been successfully defined
  list_params_not_assigned = []
  for param_name in dict_assigns:
    if not(dict_assigns[param_name]["bool_assigned"]):
      list_params_not_assigned.append(param_name)
  if len(list_params_not_assigned) == 0:
    print(f"Successfully defined turbulence driving parameters.")
  else: raise Exception("Error: failed to define the following turbulence driving parameters:", list_params_not_assigned)


## ###############################################################
## WRITE FLASH INPUT PARAMETER FILE
## ###############################################################
def writeFlashParamFile(
    filepath_ref, filepath_to, dict_sim_config, max_hours,
    str_restart = ".false.",
    str_chk_num = "0",
    str_plt_num = "0"
  ):
  ## helper function
  def _addParamAssign(param_name, param_value, comment=""):
    addParamAssign(
      dict_assigns        = dict_assigns,
      param_name          = param_name,
      param_value         = param_value,
      comment             = comment,
      nspaces_pre_assign  = FLASH_INPUT_NSPACES_PRE_ASSIGN,
      nspaces_pre_comment = FLASH_INPUT_NSPACES_PRE_COMMENT
    )
  ## initialise dictionary storing parameter assignemnts
  dict_assigns = {}
  ## add parameter assignments
  _addParamAssign(
    param_name = "cfl",
    param_value = dict_sim_config["cfl"]
  )
  _addParamAssign(
    param_name  = "st_infilename",
    param_value = FileNames.FILENAME_DRIVING_INPUT
  )
  _addParamAssign(
    param_name  = "useViscosity",
    param_value = ".true."
  )
  _addParamAssign(
    param_name  = "useMagneticResistivity",
    param_value = ".true."
  )
  _addParamAssign(
    param_name  = "diff_visc_nu",
    param_value = dict_sim_config["nu"],
    comment     = "implies Re = {} with Mach = {}".format(
      dict_sim_config["Re"],
      dict_sim_config["mach_number"]
    )
  )
  _addParamAssign(
    param_name  = "resistivity",
    param_value = dict_sim_config["eta"],
    comment     = "implies Rm = {} and Pm = {}".format(
      dict_sim_config["Rm"],
      dict_sim_config["Pm"]
    )
  )
  _addParamAssign(
    param_name  = "st_rmsMagneticField",
    param_value = dict_sim_config["init_rms_b"],
  )
  _addParamAssign(
    param_name  = "iProcs",
    param_value = dict_sim_config["num_procs"][0]
  )
  _addParamAssign(
    param_name  = "jProcs",
    param_value = dict_sim_config["num_procs"][1]
  )
  _addParamAssign(
    param_name  = "kProcs",
    param_value = dict_sim_config["num_procs"][2]
  )
  _addParamAssign(
    param_name  = "wall_clock_time_limit",
    param_value = max_hours * 60 * 60 - 1000, # [seconds]
    comment     = "close and save sim after this time has elapsed"
  )
  _addParamAssign(
    param_name  = "tmax",
    param_value = "{:.6f}".format(
      dict_sim_config["max_num_t_turb"] * dict_sim_config["t_turb"]
    ),
    comment     = "{} turb".format(dict_sim_config["max_num_t_turb"])
  )
  _addParamAssign(
    param_name  = "checkpointFileIntervalTime",
    param_value = "{:.6f}".format(dict_sim_config["t_turb"]),
    comment     = "1 t_turb"
  )
  _addParamAssign(
    param_name  = "plotFileIntervalTime",
    param_value = "{:.6f}".format(dict_sim_config["t_turb"] / dict_sim_config["outputs_per_t_turb"]),
    comment     = "1/{:d} t_turb".format(dict_sim_config["outputs_per_t_turb"])
  )
  _addParamAssign(
    param_name  = "restart",
    param_value = str_restart
  )
  _addParamAssign(
    param_name  = "checkpointFileNumber",
    param_value = str_chk_num
  )
  _addParamAssign(
    param_name  = "plotFileNumber",
    param_value = str_plt_num
  )
  ## read from reference file and write to new files
  filepath_ref  = f"{filepath_ref}/{FileNames.FILENAME_FLASH_INPUT}"
  filepath_file = f"{filepath_to}/{FileNames.FILENAME_FLASH_INPUT}"
  with open(filepath_ref, "r") as ref_file:
    with open(filepath_file, "w") as new_file:
      for ref_line in ref_file.readlines():
        new_line = processLine(
          ref_line, dict_assigns,
          nspaces_pre_assign  = FLASH_INPUT_NSPACES_PRE_ASSIGN,
          nspaces_pre_comment = FLASH_INPUT_NSPACES_PRE_COMMENT
        )
        new_file.write(new_line)
  ## check that every parameter has been successfully defined
  list_params_not_assigned = []
  for param_name in dict_assigns:
    if not(dict_assigns[param_name]["bool_assigned"]):
      list_params_not_assigned.append(param_name)
  if len(list_params_not_assigned) == 0:
    print(f"Successfully defined: {filepath_file}")
  else: raise Exception("Error: failed to define the following flash input parameters:", list_params_not_assigned)


## ###############################################################
## PREPARE ALL PARAMETER FILES FOR SIMULATION
## ###############################################################
class JobRunSim():
  def __init__(self, directory_sim, dict_sim_config, max_hours=None):
    self.directory_sim   = directory_sim
    self.dict_sim_config = dict_sim_config
    self.max_hours       = max_hours
    self._defineJobParams()
    self._createJobScript()

  def _defineJobParams(self):
    self.run_index = self.dict_sim_config["run_index"]
    self.iprocs, self.jprocs, self.kprocs = self.dict_sim_config["num_procs"]
    self.num_procs = int(self.iprocs * self.jprocs * self.kprocs)
    self.max_mem   = int(4 * self.num_procs)
    if self.max_hours is None:
      if   self.num_procs > 2.3e3: self.max_hours = 10
      elif self.num_procs > 1e3:   self.max_hours = 24
      else:                        self.max_hours = 48
    ## check group project
    if   "ek9" in self.directory_sim: self.group_project = "ek9"
    elif "jh2" in self.directory_sim: self.group_project = "jh2"
    else: raise Exception("Error: undefined group project.")
    self.job_name    = FileNames.FILENAME_RUN_SIM_JOB
    self.job_tagname = ReadFlashData.getJobTag(self.dict_sim_config, "sim")
    self.job_output  = FileNames.FILENAME_RUN_SIM_OUTPUT + str(self.run_index).zfill(2)
    self.filename_flash_exe = "flash4_nxb{}_nyb{}_nzb{}_3.0".format(
      self.dict_sim_config["num_blocks"][0],
      self.dict_sim_config["num_blocks"][1],
      self.dict_sim_config["num_blocks"][2]
    )
    k_turb = self.dict_sim_config["k_turb"]
    if k_turb < 2: raise Exception(f"Error: k_turb cannot be < 2. You have requested k_turb = {k_turb}")
    if WWFnF.checkIfFileExists(self.directory_sim, FileNames.FILENAME_DRIVING_HISTORY):
      des_amplitude = ReadFlashData.readDrivingAmplitude(self.directory_sim)
    else: des_amplitude = 0.1
    self.dict_driving_params = {
      "des_k_driv"            : k_turb,
      "des_k_min"             : k_turb - 1,
      "des_k_max"             : k_turb + 1,
      "des_velocity"          : self.dict_sim_config["mach_number"],
      "des_ampl_factor"       : des_amplitude,
      "des_sol_weight"        : 1.0,
      "des_spect_form"        : 1.0,
      "des_nsteps_per_t_turb" : 10
    }

  def _createJobScript(self):
    ## create/overwrite job file
    with open(f"{self.directory_sim}/{self.job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write(f"#PBS -P {self.group_project}\n")
      job_file.write("#PBS -q normal\n")
      job_file.write(f"#PBS -l walltime={self.max_hours}:00:00\n")
      job_file.write(f"#PBS -l ncpus={self.num_procs}\n")
      job_file.write(f"#PBS -l mem={self.max_mem}GB\n")
      job_file.write(f"#PBS -l storage=scratch/{self.group_project}+gdata/{self.group_project}\n")
      job_file.write("#PBS -l wd\n")
      job_file.write(f"#PBS -N {self.job_tagname}\n")
      job_file.write("#PBS -j oe\n")
      job_file.write("#PBS -m bea\n")
      job_file.write(f"#PBS -M neco.kriel@anu.edu.au\n")
      job_file.write("\n")
      job_file.write(". ~/modules_flash\n")
      job_file.write(f"mpirun ./{self.filename_flash_exe} 1>{self.job_output} 2>&1\n")
    ## indicate progress
    print(f"Successfully defined PBS job: {self.directory_sim}/{self.job_name}")

  def prepForRestart(self):
    ## write a new flash input file
    last_chk_filename = ReadFlashData.getLastChkFilename(self.directory_sim)
    chk_num = LoadData.readFromChkFile(f"{self.directory_sim}/{last_chk_filename}", "integer scalars", "checkpointfilenumber")
    plt_num = LoadData.readFromChkFile(f"{self.directory_sim}/{last_chk_filename}", "integer scalars", "plotfilenumber")
    writeFlashParamFile(
      filepath_ref    = FileNames.DIRECTORY_FILE_BACKUPS,
      filepath_to     = self.directory_sim,
      dict_sim_config = self.dict_sim_config,
      max_hours       = self.max_hours,
      str_restart     = ".true.",
      str_chk_num     = f"{int(chk_num):d}",
      str_plt_num     = f"{int(plt_num):d}"
    )

  def prepForRestartFromScratch(self):
    ## write a new flash input file
    writeFlashParamFile(
      filepath_ref    = FileNames.DIRECTORY_FILE_BACKUPS,
      filepath_to     = self.directory_sim,
      dict_sim_config = self.dict_sim_config,
      max_hours       = self.max_hours,
    )

  def prepFromTemplate(self):
    if not os.path.exists(FileNames.DIRECTORY_FILE_BACKUPS):
      raise Exception("Error: folder containing flash backups does not exist:", FileNames.DIRECTORY_FILE_BACKUPS)
    ## copy flash4 executable
    WWFnF.copyFile(
      directory_from = FileNames.DIRECTORY_FILE_BACKUPS,
      directory_to   = self.directory_sim,
      filename       = self.filename_flash_exe
    )
    ## write driving parameter file
    writeTurbDrivingFile(
      filepath_ref = FileNames.DIRECTORY_FILE_BACKUPS,
      filepath_to  = self.directory_sim,
      dict_params  = self.dict_driving_params
    )
    ## write flash parameter file
    writeFlashParamFile(
      filepath_ref    = FileNames.DIRECTORY_FILE_BACKUPS,
      filepath_to     = self.directory_sim,
      dict_sim_config = self.dict_sim_config,
      max_hours       = self.max_hours
    )

  def prepFromReference(self, filepath_ref_sim):
    ## copy flash4 executable
    WWFnF.copyFile(
      directory_from = FileNames.DIRECTORY_FILE_BACKUPS,
      directory_to   = self.directory_sim,
      filename       = self.filename_flash_exe
    )
    ## copy driving parameter file
    WWFnF.copyFile(
      directory_from = filepath_ref_sim,
      directory_to   = self.directory_sim,
      filename       = FileNames.FILENAME_DRIVING_INPUT
    )
    ## make sure higher resolution simulation input parameters matches the reference
    ref_sim_input = ReadFlashData.readSimConfig(filepath_ref_sim, bool_verbose=False)
    self.dict_sim_config["cfl"]                = ref_sim_input["cfl"]
    self.dict_sim_config["init_rms_b"]         = ref_sim_input["init_rms_b"]
    self.dict_sim_config["max_num_t_turb"]     = ref_sim_input["max_num_t_turb"]
    self.dict_sim_config["bool_driving_tuned"] = ref_sim_input["bool_driving_tuned"]
    ReadFlashData.saveSimConfig(self.directory_sim, self.dict_sim_config)
    print("\t> Copied key parameters from:", filepath_ref_sim)
    ## write the new flash parameter file
    writeFlashParamFile(
      filepath_ref    = FileNames.DIRECTORY_FILE_BACKUPS,
      filepath_to     = self.directory_sim,
      dict_sim_config = self.dict_sim_config,
      max_hours   = self.max_hours
    )


## END OF LIBRARY