## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

from pathlib import Path
from jormi.ww_io import io_manager
from . import _job_validation


## ###############################################################
## FUNCTIONS
## ###############################################################

def _ensure_path_is_valid(file_path: Path):
  file_path = Path(file_path).absolute()
  if file_path.suffix != ".sh": raise ValueError(f"File should end with a .sh extension: {file_path}")
  return file_path

def create_pbs_job_script(
    system_name        : str,
    directory          : str | Path,
    file_name          : str,
    command            : str,
    tag_name           : str,
    queue_name         : str,
    compute_group_name : str,
    num_procs          : int,
    wall_time_hours    : int,
    storage_group_name : str | None = None,
    email_address      : str | None = None,
    email_on_start     : bool = False,
    email_on_finish    : bool = False,
    verbose            : bool = True,
  ) -> Path:
  valid_systems = {
    "gadi": {
      "jh2": {"normal", "rsaa"},
      "ek9": {"normal", "rsaa"},
      "mk27": {"rsaa"}
    }
  }
  if queue_name not in valid_systems.get(system_name, {}).get(compute_group_name, set()):
    raise ValueError(f"Queue `{queue_name}` is not supported for compute group `{compute_group_name}` on system `{system_name}`.")
  if storage_group_name is None:
    storage_group_name = compute_group_name
  if compute_group_name != storage_group_name:
    print(f"Note: `compute_group_name` = {compute_group_name} and `storage_group_name` = {storage_group_name} are different.")
  try:
    _job_validation.validate_job_params(system_name, queue_name, num_procs, wall_time_hours)
  except _job_validation.QueueValidationError as e:
    raise ValueError(f"Invalid job parameters: {e}")
  wall_time_str = f"{wall_time_hours:02}:00:00"
  memory_limit = num_procs * 4
  mail_options = "a" # notify on failure
  if email_on_start:  mail_options += "b"
  if email_on_finish: mail_options += "e"
  file_path = io_manager.combine_file_path_parts([ directory, file_name ])
  _ensure_path_is_valid(file_path)
  with open(file_path, "w") as job_file:
    job_file.write("#!/bin/bash\n")
    job_file.write(f"#PBS -P {compute_group_name}\n")
    job_file.write(f"#PBS -q {queue_name}\n")
    job_file.write(f"#PBS -l walltime={wall_time_str}\n")
    job_file.write(f"#PBS -l ncpus={num_procs}\n")
    job_file.write(f"#PBS -l mem={memory_limit}GB\n")
    job_file.write(f"#PBS -l storage=scratch/{storage_group_name}+gdata/{storage_group_name}\n")
    job_file.write("#PBS -l wd\n")
    job_file.write(f"#PBS -N {tag_name}\n")
    job_file.write("#PBS -j oe\n")
    if email_address is not None:
      job_file.write(f"#PBS -m {mail_options}\n")
      job_file.write(f"#PBS -M {email_address}\n")
    job_file.write("\n")
    job_file.write(f'LOG_FILE="{tag_name}.out"\n')
    job_file.write('exec >"$LOG_FILE" 2>&1\n\n')
    job_file.write(f"{command.rstrip()}\n")
  if verbose:
    print("[Created PBS Job]")
    print(file_path)
    print(f"\t> Tagname  : {tag_name}")
    print(f"\t> CPUs     : {num_procs}")
    print(f"\t> Memory   : {memory_limit} GB")
    print(f"\t> Walltime : {wall_time_str}")
  return file_path


## END OF MODULE