## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import inspect
import tempfile
import traceback
from tqdm import tqdm
from typing import Callable, Tuple, Any, List, Optional
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def _func_wrapper(
  func       : Callable,
  func_args  : Tuple[Any, ...],
  debug_mode : bool = False,
  plot_latex : bool = False
) -> Any:
  """
  Wraps a function call with optional LaTeX plotting setup and exception handling.
  Sets up a temporary matplotlib environment if plot_latex is True.
  """
  original_env = {}
  tmp_directory = None
  try:
    if plot_latex:
      ## set up temporary directory for matplotlib and LaTeX configs
      tmp_directory = tempfile.TemporaryDirectory(prefix="mpl_")
      original_env.update({
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
        "TEXMFVAR": os.environ.get("TEXMFVAR")
      })
      os.environ["MPLCONFIGDIR"] = tmp_directory.name
      os.environ["TEXMFVAR"] = os.path.join(tmp_directory.name, "texmfvar")
      import matplotlib
      matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    return _execute_func(func, func_args, debug_mode)
  except Exception as e:
    return {"success": False, "error": _extract_exception(e)}
  finally:
    ## clean up and restore original environment
    if plot_latex and tmp_directory:
      _restore_mpl_env(original_env)
      tmp_directory.cleanup()

def _execute_func(
  func       : Callable,
  func_args  : Tuple[Any, ...],
  debug_mode : bool,
) -> Any:
  """Executes the function with provided arguments. Injects 'debug_mode' if accepted by the function signature."""
  try:
    signature = inspect.signature(func)
    kwargs = {}
    if "debug_mode" in signature.parameters:
      kwargs["debug_mode"] = debug_mode
    return {
      "success" : True,
      "result"  : func(*func_args, **kwargs)
    }
  except Exception as e:
    return {
      "success" : False,
      "error"   : _extract_exception(e)
    }

def _extract_exception(e: Exception) -> str:
  """Returns formatted exception string with traceback details."""
  return f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"

def _restore_mpl_env(original_env: dict) -> None:
  """Restores original matplotlib and LaTeX-related environment variables."""
  for var in ["MPLCONFIGDIR", "TEXMFVAR"]:
    if original_env.get(var):
      os.environ[var] = original_env[var]
    else:
      os.environ.pop(var, None)

def _print_failures(failed_tasks: List[tuple]) -> None:
  """Prints formatted error messages for all failed tasks."""
  print(f"\n{len(failed_tasks)} TASK(S) FAILED:")
  for task_index, task_error in failed_tasks:
    print(f"\n--- Task {task_index} Error ---")
    print(str(task_error).strip())


## ###############################################################
## OPERATOR FUNCTION
## ###############################################################
def run_in_parallel(
  func,
  args_list,
  *,
  num_procs       = None,
  timeout_seconds = None,
  show_progress   = True,
  show_failures   = False
):
  task_results = [None] * len(args_list)
  failed_tasks = []
  with ProcessPool(max_workers=num_procs) as pool:
    pending_tasks = [
      (i, pool.schedule(func, args=args, timeout=timeout_seconds))
      for i, args in enumerate(args_list)
    ]
    if show_progress:
      pending_tasks = tqdm(pending_tasks, total=len(pending_tasks), desc="Processing", unit="task")
    for i, future in pending_tasks:
      try:
        task_results[i] = future.result()  # Will raise TimeoutError if timed out
      except TimeoutError:
        error_message = f"TimeoutError: Task {i} timed out after {timeout_seconds}s"
        task_results[i] = None
        failed_tasks.append((i, error_message))
      except ProcessExpired as error:
        error_message = f"ProcessExpired: Task {i} exited with code {error.exitcode}"
        task_results[i] = None
        failed_tasks.append((i, error_message))
      except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        task_results[i] = None
        failed_tasks.append((i, error_message))
  if failed_tasks and show_failures:
    error_summary = "\n".join([
      f"Task {task_index} failed: {str(error_message).splitlines()[0]}"
      for task_index, error_message in failed_tasks
    ])
    raise RuntimeError(f"{len(failed_tasks)} tasks failed:\n{error_summary}")
  return [result for result in task_results if result is not None]


## END OF MODULE