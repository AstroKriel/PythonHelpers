## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
from tqdm import tqdm
from typing import Callable, Tuple, Any, Optional
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError


## ###############################################################
## OPERATOR FUNCTION
## ###############################################################
def run_in_parallel(
  func      : Callable,
  args_list : Tuple[Any, ...],
  *,
  num_procs       : Optional[int] = None,
  timeout_seconds : Optional[float] = None,
  show_progress   : bool = True,
):
  task_results = [None] * len(args_list)
  failed_tasks = []
  with ProcessPool(max_workers=num_procs) as pool:
    pending_tasks = [
      (task_index, pool.schedule(func, args=task_args, timeout=timeout_seconds))
      for task_index, task_args in enumerate(args_list)
    ]
    if show_progress:
      pending_tasks = tqdm(pending_tasks, total=len(pending_tasks), desc="Processing", unit="task")
    for task_index, task in pending_tasks:
      try:
        task_results[task_index] = task.result()
      except TimeoutError:
        error_message = f"TimeoutError: Task {task_index} timed out after {timeout_seconds}s"
        task_results[task_index] = None
        failed_tasks.append((task_index, error_message))
      except ProcessExpired as error:
        error_message = f"ProcessExpired: Task {task_index} exited with code {error.exitcode}"
        task_results[task_index] = None
        failed_tasks.append((task_index, error_message))
      except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        task_results[task_index] = None
        failed_tasks.append((task_index, error_message))
  if failed_tasks:
    error_summary = "\n".join([
      f"Task {task_index} failed: {str(error_message).splitlines()[0]}"
      for task_index, error_message in failed_tasks
    ])
    raise RuntimeError(f"{len(failed_tasks)} tasks failed:\n{error_summary}")
  return [result for result in task_results if result is not None]


## END OF MODULE