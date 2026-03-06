## { TEST

##
## === DEPENDENCIES
##

import os
import shutil
import time
import numpy
import unittest

from typing import Any
from collections.abc import Callable
from pathlib import Path

from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, annotate_axis
from jormi.utils import parallel_utils

##
## === WORKER FUNCTIONS
##


def dummy_task(
    arg: Any,
) -> Any:
    return arg


def cpu_heavy_task(
    block_of_values: list[float],
) -> float:
    total = 0.0
    for _ in range(200):
        for value in block_of_values:
            total += value**1.00001
    return total


def time_fn(
    worker_fn: Callable[..., Any],
    grouped_args: list[Any],
    num_repeats: int,
    num_workers: int,
    verbose: bool = True,
) -> float:
    elapsed_times = []
    for _ in range(num_repeats):
        start_time = time.perf_counter()
        parallel_utils.run_in_parallel(
            worker_fn=worker_fn,
            grouped_args=grouped_args,
            num_workers=num_workers,
            show_progress=False,
        )
        elapsed_times.append(time.perf_counter() - start_time)
    ave_elapsed_time = numpy.median(elapsed_times)
    std_elapsed_time = numpy.std(elapsed_times)
    if verbose:
        print(
            f"{num_workers:d} procs completed in {ave_elapsed_time:.3f} ± {std_elapsed_time:.3f} seconds.",
        )
    return float(ave_elapsed_time)


def sleepy_task(
    duration: float,
) -> None:
    time.sleep(duration)


def error_task() -> None:
    raise ValueError("Simulated error")


def mixed_task(
    value: float,
) -> float:
    return value / (value - 5)


def crashing_task() -> None:
    os._exit(1)


def delayed_return(
    duration: float,
    result: Any,
) -> Any:
    time.sleep(duration)
    return result


def plot_task(
    fig_directory: Path,
    num_samples: int,
) -> bool:
    try:
        fig, ax = plot_manager.create_figure()
        x_values = numpy.linspace(0, 5 * numpy.pi, num_samples)
        y_values = numpy.sin(x_values)
        ax.plot(x_values, y_values, color="black", ls="-", lw=1, marker="o", ms=5)
        ax.set_xlabel(r"$\sum_{\forall i}x_{i}^{2}$")
        ax.set_ylabel(r"$\sin(2\pi x + 32)$")
        annotate_axis.add_text(ax, 0.05, 0.95, r"$(0.05, 0.95)$ \% of the fig uniform_domain")
        fig_name = f"plot_with_{(num_samples):04d}_samples.png"
        fig_path = io_manager.combine_file_path_parts([fig_directory, fig_name])
        plot_manager.save_figure(fig, fig_path, verbose=False)
        return True
    except:
        return False


class TestParallelExecution(unittest.TestCase):

    def test_parallel_plotting(self):
        script_directory = io_manager.get_caller_directory()
        fig_directory = io_manager.combine_file_path_parts([script_directory, "plots"])
        io_manager.init_directory(fig_directory, verbose=False)
        self.addCleanup(shutil.rmtree, fig_directory, True)
        grouped_args = [(
            fig_directory,
            5 + 5 * plot_index,
        ) for plot_index in range(100)]
        result = parallel_utils.run_in_parallel(
            worker_fn=plot_task,
            grouped_args=grouped_args,
            show_progress=False,
        )
        self.assertEqual(all(result), True)

    def test_timeout(self):
        grouped_args = [(duration, ) for duration in [0.5, 1, 3, 5]]
        try:
            parallel_utils.run_in_parallel(
                worker_fn=sleepy_task,
                grouped_args=grouped_args,
                timeout_seconds=1.5,
                num_workers=2,
                show_progress=False,
            )
            self.fail("Expected a RuntimeError due to timeout, but none was raised.")
        except RuntimeError as runtime_error:
            self.assertIn("tasks failed", str(runtime_error))
            self.assertNotIn("Task 0 timed out", str(runtime_error))
            self.assertNotIn("Task 1 timed out", str(runtime_error))
            self.assertIn("Task 2 timed out", str(runtime_error))
            self.assertIn("Task 3 timed out", str(runtime_error))
            self.assertNotIn("Task 4 timed out", str(runtime_error))

    def test_parallel_scaling(self):
        ## compare 1-worker vs max-workers
        num_values_per_block = 1000
        num_blocks = 64
        blocks = [[float(value) for value in range(num_values_per_block)] for _ in range(num_blocks)]
        grouped_args = [(block_of_values, ) for block_of_values in blocks]
        min_speedup_factor = 1.5
        max_workers = min(8, os.cpu_count() or 1)
        if max_workers < 2:
            self.skipTest("Need at least 2 CPUs for scaling test.")
        time_serial = time_fn(
            worker_fn=cpu_heavy_task,
            grouped_args=grouped_args,
            num_repeats=3,
            num_workers=1,
            verbose=False,
        )
        time_parallel = time_fn(
            worker_fn=cpu_heavy_task,
            grouped_args=grouped_args,
            num_repeats=3,
            num_workers=max_workers,
            verbose=False,
        )
        self.assertGreater(time_serial, time_parallel * min_speedup_factor)

    def test_parallel_correctness(self):
        num_values_per_block = 10
        num_blocks = 6
        blocks = [[float(value) for value in range(num_values_per_block)] for _ in range(num_blocks)]
        grouped_args = [(block_of_values, ) for block_of_values in blocks]
        expected_results = [cpu_heavy_task(block_of_values) for block_of_values in blocks]
        results = parallel_utils.run_in_parallel(
            worker_fn=cpu_heavy_task,
            grouped_args=grouped_args,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(len(results), len(expected_results))
        for result, expected in zip(results, expected_results):
            self.assertEqual(result, expected)

    def test_empty_grouped_args(self):
        grouped_args = []
        result = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_args=grouped_args,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(result, [])

    def test_exception_propagation(self):
        grouped_args = [()] * 3
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=error_task,
                grouped_args=grouped_args,
                num_workers=2,
                show_progress=False,
            )
        error_lines = str(cm.exception).split('\n')[1:]
        self.assertEqual(len(error_lines), 3)
        self.assertTrue(all("ValueError" in line for line in error_lines))

    def test_mixed_success_failure(self):
        grouped_args = [(task_index, ) for task_index in range(10)]
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=mixed_task,
                grouped_args=grouped_args,
                num_workers=2,
                show_progress=False,
            )
        error = cm.exception
        self.assertIn("Task 5 failed", str(error))
        self.assertIn("Task 5 failed: ZeroDivisionError", str(error))

    def test_process_expiry_handling(self):
        grouped_args = [()] * 3
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=crashing_task,
                grouped_args=grouped_args,
                num_workers=2,
                show_progress=False,
            )
        self.assertIn("ProcessExpired", str(cm.exception))

    def test_result_ordering(self):
        grouped_args = [(0.2, 3), (10.1, 1), (0.3, 4), (0.0, 2)]
        results = parallel_utils.run_in_parallel(
            worker_fn=delayed_return,
            grouped_args=grouped_args,
            num_workers=4,
            show_progress=False,
        )
        self.assertEqual(results, [3, 1, 4, 2])

    def test_various_data_types(self):
        grouped_args = [
            ("hello", ),
            ({
                "key": "value",
            }, ),
            (123, ),
            (b"bytes", ),
        ]
        results = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_args=grouped_args,
            num_workers=2,
            show_progress=False,
        )
        expected_results = [args[0] for args in grouped_args]
        self.assertEqual(results, expected_results)

    def test_scalar_arg_normalisation(self):
        ## scalar args (non-list, non-tuple) should be wrapped into single-element lists
        grouped_args = [1, 2, 3]
        results = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_args=grouped_args,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(results, [1, 2, 3])

    def test_show_progress_does_not_crash(self):
        ## show_progress=True wraps iteration in tqdm; verify it doesn't alter results
        grouped_args = [(task_index, ) for task_index in range(4)]
        results = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_args=grouped_args,
            num_workers=2,
            show_progress=True,
        )
        self.assertEqual(results, [0, 1, 2, 3])

    def test_default_num_workers(self):
        ## num_workers=None should fall back to os.cpu_count() without error
        grouped_args = [(task_index, ) for task_index in range(4)]
        results = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_args=grouped_args,
            num_workers=None,
            show_progress=False,
        )
        self.assertEqual(results, [0, 1, 2, 3])

    def test_no_timeout(self):
        ## timeout_seconds=None should not raise for tasks that take some time
        grouped_args = [(0.1, 2), (0.1, 3)]
        results = parallel_utils.run_in_parallel(
            worker_fn=delayed_return,
            grouped_args=grouped_args,
            timeout_seconds=None,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(results, [2, 3])


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
