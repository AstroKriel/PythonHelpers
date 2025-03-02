## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import shlex
import subprocess

## load user defined modules
from Loki.TheUsefulModule import WWFnF


## ###############################################################
## FUNCTIONS
## ###############################################################
def checkIfShellPrivilegesAreRequired(command):
  dict_shell_syntax = {
    "|",      # pipe: `ls | grep .py`
    # "&",      # background execution: `sleep 5 &`
    # ";",      # command separator: `cmd1; cmd2`
    "<",      # input redirection: `sort < file.txt`
    ">",      # output redirection: `echo "hi" > file.txt`
    # "(", ")", # subshell execution: `(cd /tmp && ls)`
    "$",      # variable expansion: `echo $HOME`
    "*", "?", # wildcards: `rm *.txt`, `ls file?.txt`
    # "#",      # comment: `echo hello # ignored part`
    # "{", "}", # brace expansion: `echo {a,b,c}`
    # "=",      # variable assignment: `VAR=value`
    # "[", "]", # test conditions: `[ -f file.txt ]`
    # "~",      # home directory: `cd ~`
  }
  return any(
    char in dict_shell_syntax
    for char in command
  )

def runCommand(
    command,
    directory = None,
    timeout   = None,
    bool_capture_output = True,
  ):
  bool_shell_required = checkIfShellPrivilegesAreRequired(command)
  try:
    result = subprocess.run(
      command if bool_shell_required else shlex.split(command),
      cwd            = directory,
      timeout        = timeout,
      capture_output = bool_capture_output,
      shell          = bool_shell_required,
      check          = False,
      text           = True,
    )
  except FileNotFoundError as exception:
    raise RuntimeError(f"Command `{command}` could not be executed.") from exception
  except subprocess.TimeoutExpired as exception:
    raise RuntimeError(f"Command `{command}` timed out after `{timeout}` seconds.") from exception
  if bool_capture_output and (result.returncode != 0):
    message = f"The following command failed with return code `{result.returncode}`: {command}"
    if result.stdout: message += f"\nstdout: {result.stdout.strip()}"
    if result.stderr: message += f"\nstderr: {result.stderr.strip()}"
    raise RuntimeError(message)
  return result.stdout if bool_capture_output else result


## END OF LIBRARY