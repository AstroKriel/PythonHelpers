## { MODULE

##
## === DEPENDENCIES
##

import os
import multiprocessing
from tqdm import tqdm
from typing import Callable, Any, Iterable, List
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError

##
## === OPERATOR FUNCTIONS
##


def _spawn_fresh_processes() -> None:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _normalise_grouped_args(grouped_worker_args: Iterable[Any]) -> List[List[Any]]:
    _grouped_args: List[List[Any]] = []
    for args in grouped_worker_args:
        if isinstance(args, (list, tuple)) and not isinstance(args, (str, bytes)):
            _grouped_args.append(list(args))
        else:
            _grouped_args.append([args])
    return _grouped_args


def _enable_plotting(
    theme: str = "light",
    use_tex: bool = True,
) -> None:
    import os, tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_cfg_"))
    os.environ.setdefault("TEXMFOUTPUT", tempfile.mkdtemp(prefix="mpl_tex_"))
    import matplotlib
    matplotlib.use("Agg", force=True)
    from jormi.ww_plots.plot_styler import set_theme
    set_theme(theme=theme, use_tex=use_tex)


def _invoke_with_plotting(
    worker_fn: Callable[..., Any],
    task_args: List[Any],
    theme: str = "light",
    use_tex: bool = True,
) -> Any:
    _enable_plotting(theme=theme, use_tex=use_tex)
    return worker_fn(*task_args)


def run_in_parallel(
    *,
    worker_fn: Callable[..., Any],
    grouped_worker_args: Iterable[Any],
    num_workers: int | None = None,
    timeout_seconds: float | None = None,
    show_progress: bool = True,
    enable_plotting: bool = False,
    theme: str = "light",
    use_tex: bool = True,
) -> List[Any]:
    _spawn_fresh_processes()
    grouped_worker_args = _normalise_grouped_args(grouped_worker_args)
    if num_workers is None: num_workers = os.cpu_count() or 1
    task_results: list[Any] = [None] * len(grouped_worker_args)
    failed_tasks: list[tuple[int, str]] = []
    with ProcessPool(max_workers=num_workers) as pool:
        pending_tasks = []
        for task_index, task_args in enumerate(grouped_worker_args):
            if enable_plotting:
                task_pair = [worker_fn, task_args]
                task = (
                    pool.schedule(
                        _invoke_with_plotting,
                        args=task_pair,
                        timeout=timeout_seconds,
                        kwargs={
                            "theme": theme,
                            "use_tex": use_tex,
                        },
                    ) if timeout_seconds is not None else pool.schedule(
                        _invoke_with_plotting,
                        args=task_pair,
                        kwargs={
                            "theme": theme,
                            "use_tex": use_tex,
                        },
                    )
                )
            else:
                task = (
                    pool.schedule(worker_fn, args=task_args, timeout=timeout_seconds)
                    if timeout_seconds is not None else pool.schedule(worker_fn, args=task_args)
                )
            pending_tasks.append((task_index, task))
        if show_progress:
            pending_tasks = tqdm(
                pending_tasks,
                total=len(pending_tasks),
                desc="Processing",
                unit="tasks",
            )
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
        error_summary = "\n".join(
            [
                f"Task {task_index} failed: {str(error_message).splitlines()[0]}"
                for task_index, error_message in failed_tasks
            ],
        )
        raise RuntimeError(f"{len(failed_tasks)} tasks failed:\n{error_summary}")
    return task_results


## } MODULE
