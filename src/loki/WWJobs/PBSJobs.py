## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
from loki.WWIO import IOShell
from loki.Utils import Utils4IO


## ###############################################################
## FUNCTIONS
## ###############################################################
def submit_job(directory, job_name, bool_check_job_status=False) -> bool:
  if bool_check_job_status and is_job_already_in_queue(directory, job_name):
    print("Job is already currently running:", job_name)
    return False
  print("Submitting job:", job_name)
  try:
    IOShell.execute_shell_command(f"qsub {job_name}", directory=directory, bool_force_shell=True)
    return True
  except RuntimeError as e:
    print(f"Failed to submit job `{job_name}`: {e}")
    return False

def is_job_already_in_queue(directory, job_filename):
  """Checks if a job name is already in the queue."""
  if not Utils4IO.does_file_exist(directory, job_filename):
    print(f"`{job_filename}` job file does not exist in: {directory}")
    return False
  job_tagname = get_job_name_from_pbs_script(directory, job_filename)
  if not job_tagname:
    print(f"Error: `#PBS -N` not found in job file: {job_filename}")
    return False
  list_job_tagnames = get_list_of_queued_jobs()
  if list_job_tagnames is None: return False
  return job_tagname in list_job_tagnames

def get_job_name_from_pbs_script(directory, job_filename):
  """Gets the job name from a PBS job script."""
  with open(Utils4IO.create_file_path(directory, job_filename), "r") as fp:
    for line in fp:
      if "#PBS -N" in line:
        return line.strip().split(" ")[-1] if line.strip() else None
  return None

def get_list_of_queued_jobs():
  """Collects all job names currently in the queue."""
  try:
    output = IOShell.execute_shell_command("qstat -f | grep Job_Name", bool_capture_output=True)
    return {
        line.strip().split()[-1]
        for line in output.split("\n") if line.strip()
    } if output.strip() else set()
  except RuntimeError as e:
    print(f"Error retrieving job names from the queue: {e}")
    return None



## END OF MODULE