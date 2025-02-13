## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import subprocess

## load user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## HELPFUL FUNCTIONS
## ###############################################################
def printLine(mssg):
  if isinstance(mssg, list): print(*mssg, flush=True)
  else: print(mssg, flush=True)

def runCommand(
    command,
    directory  = None,
    bool_debug = False
  ):
  if bool_debug: print(command)
  else:
    p = subprocess.Popen(command, shell=True, cwd=directory)
    p.wait()

def getCommandOutput(command, directory=None):
  p = subprocess.run(command, shell=True, cwd=directory, stdout=subprocess.PIPE)
  return p.stdout.decode("utf-8")

def submitJob(directory, job_name, bool_ignore_job=False):
  if not(bool_ignore_job) and WWFnF.checkIfJobIsRunning(directory, job_name):
    print("Job is already currently running:", job_name)
  else:
    print("Submitting job:", job_name)
    runCommand(f"qsub {job_name}", directory=directory)


## END OF LIBRARY