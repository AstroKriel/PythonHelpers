## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import subprocess
from Loki.WWIO import Terminal
from Loki.WWIO import FileFolderIO


## ###############################################################
## FUNCTIONS
## ###############################################################
def submitJob(directory, job_name, bool_check_job_status=False):
  if bool_check_job_status and checkIfJobIsInQueue(directory, job_name):
    print("Job is already currently running:", job_name)
  else:
    print("Submitting job:", job_name)
    Terminal.runCommand(f"qsub {job_name}", directory=directory)


def checkIfJobIsInQueue(directory, job_filename):
  if not FileFolderIO.checkIfFileExists(directory, job_filename):
    print(f"Note: `{job_filename}` job file does not exist in: {directory}")
    return False
  try:
    list_job_tagnames = Terminal.getCommandOutput("qstat -f | grep Job_Name")
  except subprocess.CalledProcessError as e:
    print(f"Error retrieving job names from the queue: {e}")
    return False
  job_tagname = None
  with open(FileFolderIO.createFilepathString(directory, job_filename), "r") as fp:
    for line in fp.readlines():
      if "#PBS -N" in line:
        job_tagname = line.split(" ")[-1]
        break
  if not job_tagname:
    print(f"Error: `#PBS -N` not found in job file: {job_filename}")
    return False
  return job_tagname in list_job_tagnames


## END OF MODULE