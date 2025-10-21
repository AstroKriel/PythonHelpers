import os
import time
import numpy
import unittest
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, annotate_axis
from jormi.utils import parallel_utils


def dummy_task(arg):
    return arg


def cpu_heavy_task(block_of_values):
    total = 0.0
    for _ in range(200):
        for value in block_of_values:
            total += value**1.00001
    return total


def time_fn(worker_fn, grouped_worker_args, num_repeats, num_workers, verbose=True):
    elapsed_times = []
    for _ in range(num_repeats):
        start_time = time.perf_counter()
        parallel_utils.run_in_parallel(
            worker_fn=worker_fn,
            grouped_worker_args=grouped_worker_args,
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
    return ave_elapsed_time


def sleepy_task(duration):
    time.sleep(duration)


def error_task():
    raise ValueError("Simulated error")


def mixed_task(x):
    return x / (x - 5)


def crashing_task():
    os._exit(1)


def delayed_return(duration, value):
    time.sleep(duration)
    return value


def plot_task(fig_directory, num_samples):
    try:
        fig, axs_grid = plot_manager.create_figure()
        ax = axs_grid[0, 0]
        x = numpy.linspace(0, 5 * numpy.pi, num_samples)
        y = numpy.sin(x)
        ax.plot(x, y, color="black", ls="-", lw=1, marker="o", ms=5)
        ax.set_xlabel(r"$\sum_{\forall i}x_{i}^{2}$")
        ax.set_ylabel(r"$\sin(2\pi x + 32)$")
        annotate_axis.add_text(ax, 0.05, 0.95, r"$(0.05, 0.95)$ \% of the fig uniform_domain")
        fig_name = f"plot_with_{(num_samples):04d}_samples.png"
        fig_file_path = io_manager.combine_file_path_parts([fig_directory, fig_name])
        plot_manager.save_figure(fig, fig_file_path, verbose=False)
        return True
    except:
        return False


class TestParallelExecution(unittest.TestCase):

    def test_parallel_plotting(self):
        script_directory = io_manager.get_caller_directory()
        fig_direcotory = io_manager.combine_file_path_parts([script_directory, "plots"])
        io_manager.init_directory(fig_direcotory, verbose=False)
        grouped_worker_args = [(
            fig_direcotory,
            5 + 5 * plot_index,
        ) for plot_index in range(100)]
        result = parallel_utils.run_in_parallel(
            worker_fn=plot_task,
            grouped_worker_args=grouped_worker_args,
            # num_workers     = 8,
            show_progress=False,
        )
        self.assertEqual(all(result), True)

    def test_timeout(self):
        grouped_worker_args = [(d, ) for d in [0.5, 1, 3, 5]]
        try:
            parallel_utils.run_in_parallel(
                worker_fn=sleepy_task,
                grouped_worker_args=grouped_worker_args,
                timeout_seconds=1.5,
                num_workers=2,
                show_progress=False,
            )
            self.fail("Expected a RuntimeError due to timeout, but none was raised.")
        except RuntimeError as e:
            self.assertIn("tasks failed", str(e))
            self.assertNotIn("Task 0 timed out", str(e))
            self.assertNotIn("Task 1 timed out", str(e))
            self.assertIn("Task 2 timed out", str(e))
            self.assertIn("Task 3 timed out", str(e))
            self.assertNotIn("Task 4 timed out", str(e))

    def test_parallel_scaling(self):
        num_values_per_block = 1000
        num_blocks = 64
        blocks = [[float(x) for x in range(num_values_per_block)] for _ in range(num_blocks)]
        grouped_worker_args = [(block_of_values, ) for block_of_values in blocks]
        elapsed_times = []
        for num_workers in [1, 2, 4, 8]:
            ave_elapsed_time = time_fn(
                worker_fn=cpu_heavy_task,
                grouped_worker_args=grouped_worker_args,
                num_repeats=5,
                num_workers=num_workers,
                verbose=False,
            )
            elapsed_times.append(ave_elapsed_time)
        for pair_index in range(len(elapsed_times) - 1):
            self.assertGreater(elapsed_times[pair_index], elapsed_times[pair_index + 1])

    def test_parallel_correctness(self):
        num_values_per_block = 10
        num_blocks = 6
        blocks = [[float(x) for x in range(num_values_per_block)] for _ in range(num_blocks)]
        grouped_worker_args = [(block_of_values, ) for block_of_values in blocks]
        expected_results = [cpu_heavy_task(block_of_values) for block_of_values in blocks]
        results = parallel_utils.run_in_parallel(
            worker_fn=cpu_heavy_task,
            grouped_worker_args=grouped_worker_args,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(len(results), len(expected_results))
        for result, expected in zip(results, expected_results):
            self.assertEqual(result, expected)

    def test_empty_grouped_args(self):
        grouped_worker_args = []
        result = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_worker_args=grouped_worker_args,
            num_workers=2,
            show_progress=False,
        )
        self.assertEqual(result, [])

    def test_exception_propagation(self):
        grouped_worker_args = [()] * 3
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=error_task,
                grouped_worker_args=grouped_worker_args,
                num_workers=2,
                show_progress=False,
            )
        error_lines = str(cm.exception).split('\n')[1:]
        self.assertEqual(len(error_lines), 3)
        self.assertTrue(all("ValueError" in line for line in error_lines))

    def test_mixed_success_failure(self):
        grouped_worker_args = [(i, ) for i in range(10)]
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=mixed_task,
                grouped_worker_args=grouped_worker_args,
                num_workers=2,
                show_progress=False,
            )
        error = cm.exception
        self.assertIn("Task 5 failed", str(error))
        self.assertIn("Task 5 failed: ZeroDivisionError", str(error))

    def test_process_expiry_handling(self):
        grouped_worker_args = [()] * 3
        with self.assertRaises(RuntimeError) as cm:
            parallel_utils.run_in_parallel(
                worker_fn=crashing_task,
                grouped_worker_args=grouped_worker_args,
                num_workers=2,
                show_progress=False,
            )
        self.assertIn("ProcessExpired", str(cm.exception))

    def test_result_ordering(self):
        grouped_worker_args = [(0.2, 3), (10.1, 1), (0.3, 4), (0.0, 2)]
        results = parallel_utils.run_in_parallel(
            worker_fn=delayed_return,
            grouped_worker_args=grouped_worker_args,
            num_workers=4,
            show_progress=False,
        )
        self.assertEqual(results, [3, 1, 4, 2])

    def test_various_data_types(self):
        grouped_worker_args = [
            ("hello", ),
            ({
                "key": "value",
            }, ),
            (123, ),
            (b"bytes", ),
        ]
        results = parallel_utils.run_in_parallel(
            worker_fn=dummy_task,
            grouped_worker_args=grouped_worker_args,
            num_workers=2,
            show_progress=False,
        )
        expected_reults = [arg[0] for arg in grouped_worker_args]
        self.assertEqual(results, expected_reults)


if __name__ == "__main__":
    unittest.main()
