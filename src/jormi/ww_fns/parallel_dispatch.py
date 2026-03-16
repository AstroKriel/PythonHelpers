## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import multiprocessing
import os

from concurrent.futures import TimeoutError
from typing import (
    Any,
    Callable,
    Iterable,
)

## third-party
from pebble import (
    ProcessExpired,
    ProcessPool,
)
from tqdm import tqdm

##
## === HELPERS
##


def _spawn_fresh_processes() -> None:
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _normalise_grouped_args(
    grouped_args: Iterable[Any],
) -> list[list[Any]]:
    _grouped_args: list[list[Any]] = []
    for args in grouped_args:
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
    from jormi.ww_plots.style_plots import set_theme
    set_theme(
        theme=theme,
        use_tex=use_tex,
    )


def _invoke_with_plotting(
    worker_fn: Callable[..., Any],
    task_args: list[Any],
    theme: str = "light",
    use_tex: bool = True,
) -> Any:
    _enable_plotting(
        theme=theme,
        use_tex=use_tex,
    )
    return worker_fn(*task_args)


##
## === OPERATOR FUNCTIONS
##


def run_in_parallel(
    *,
    worker_fn: Callable[..., Any],
    grouped_args: Iterable[Any],
    num_workers: int | None = None,
    timeout_seconds: float | None = None,
    show_progress: bool = True,
    enable_plotting: bool = False,
    theme: str = "light",
    use_tex: bool = True,
) -> list[Any]:
    _spawn_fresh_processes()
    grouped_args = _normalise_grouped_args(grouped_args)
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    task_results: list[Any] = [None] * len(grouped_args)
    failed_tasks: list[tuple[int, str]] = []
    with ProcessPool(max_workers=num_workers) as pool:
        tasks = []
        for task_index, task_args in enumerate(grouped_args):
            if enable_plotting:
                task = pool.schedule(
                    _invoke_with_plotting,
                    args=[worker_fn, task_args],
                    timeout=timeout_seconds, # type: ignore[arg-type]
                    kwargs={
                        "theme": theme,
                        "use_tex": use_tex,
                    },
                )
            else:
                task = pool.schedule(
                    function=worker_fn,
                    args=task_args,
                    timeout=timeout_seconds,  # type: ignore[arg-type]
                )
            tasks.append((task_index, task))
        iterator = tqdm(tasks, total=len(tasks), desc="Processing", unit="tasks") if show_progress else tasks
        for task_index, task in iterator:
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
