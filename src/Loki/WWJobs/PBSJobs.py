## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import subprocess
from Loki.WWIO import IOShell
from Loki.WWIO import IOFnF


## ###############################################################
## FUNCTIONS
## ###############################################################
def submitJob(directory, job_name, bool_check_job_status=False) -> bool:
  if bool_check_job_status and checkIfJobIsInQueue(directory, job_name):
    print("Job is already currently running:", job_name)
    return False
  print("Submitting job:", job_name)
  try:
    IOShell.runCommand(f"qsub {job_name}", directory=directory, bool_force_shell=True)
    return True
  except RuntimeError as e:
    print(f"Failed to submit job `{job_name}`: {e}")
    return False

def checkIfJobIsInQueue(directory, job_filename):
  """Checks if a job name is already in the queue."""
  if not IOFnF.checkIfFileExists(directory, job_filename):
    print(f"`{job_filename}` job file does not exist in: {directory}")
    return False
  job_tagname = getJobName(directory, job_filename)
  if not job_tagname:
    print(f"Error: `#PBS -N` not found in job file: {job_filename}")
    return False
  list_job_tagnames = getQueuedJobNames()
  if list_job_tagnames is None: return False
  return job_tagname in list_job_tagnames

def getJobName(directory, job_filename):
  """Gets the job name from a PBS job script."""
  with open(IOFnF.createFilepathString(directory, job_filename), "r") as fp:
    for line in fp:
      if "#PBS -N" in line:
        return line.strip().split(" ")[-1] if line.strip() else None
  return None

def getQueuedJobNames():
  """Collects all job names currently in the queue."""
  try:
    output = IOShell.runCommand("qstat -f | grep Job_Name", bool_capture_output=True)
    return {
        line.strip().split()[-1]
        for line in output.split("\n") if line.strip()
    } if output.strip() else set()
  except RuntimeError as e:
    print(f"Error retrieving job names from the queue: {e}")
    return None



## END OF MODULE