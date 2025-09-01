import os
import time
import numpy
import unittest
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, add_annotations
from jormi.parallelism import independent_tasks


def dummy_task(arg):
    return arg


def cpu_heavy_task(block_of_values):
    total = 0.0
    for _ in range(200):
        for value in block_of_values:
            total += value**1.00001
    return total


def time_function(func, args_list, num_repeats, num_procs, verbose=True):
    elapsed_times = []
    for _ in range(num_repeats):
        start_time = time.perf_counter()
        independent_tasks.run_in_parallel(
            func=func,
            args_list=args_list,
            num_procs=num_procs,
            show_progress=False,
        )
        elapsed_times.append(time.perf_counter() - start_time)
    ave_elapsed_time = numpy.median(elapsed_times)
    std_elapsed_time = numpy.std(elapsed_times)
    if verbose:
        print(
            f"{num_procs:d} procs completed in {ave_elapsed_time:.3f} ± {std_elapsed_time:.3f} seconds.",
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
        fig, ax = plot_manager.create_figure()
        x = numpy.linspace(0, 5 * numpy.pi, num_samples)
        y = numpy.sin(x)
        ax.plot(x, y, color="black", ls="-", lw=1, marker="o", ms=5)
        ax.set_xlabel(r"$\sum_{\forall i}x_{i}^{2}$")
        ax.set_ylabel(r"$\sin(2\pi x + 32)$")
        add_annotations.add_text(ax, 0.05, 0.95, r"$(0.05, 0.95)$ \% of the fig domain")
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
        args_list = [(
            fig_direcotory,
            5 + 5 * plot_index,
        ) for plot_index in range(100)]
        result = independent_tasks.run_in_parallel(
            func=plot_task,
            args_list=args_list,
            # num_procs     = 8,
            show_progress=False,
        )
        self.assertEqual(all(result), True)

    # def test_timeout(self):
    #   args_list = [(d,) for d in [0.5, 1, 3, 5]]
    #   try:
    #     independent_tasks.run_in_parallel(
    #       func            = sleepy_task,
    #       args_list       = args_list,
    #       timeout_seconds = 1.5,
    #       num_procs       = 2,
    #       show_progress   = False
    #     )
    #     self.fail("Expected a RuntimeError due to timeout, but none was raised.")
    #   except RuntimeError as e:
    #     self.assertIn("tasks failed", str(e))
    #     self.assertNotIn("Task 0 timed out", str(e))
    #     self.assertNotIn("Task 1 timed out", str(e))
    #     self.assertIn("Task 2 timed out", str(e))
    #     self.assertIn("Task 3 timed out", str(e))
    #     self.assertNotIn("Task 4 timed out", str(e))

    # def test_parallel_scaling(self):
    #   num_values_per_block = 1000
    #   num_blocks = 64
    #   blocks = [
    #     [
    #       float(x)
    #       for x in range(num_values_per_block)
    #     ] for _ in range(num_blocks)
    #   ]
    #   args_list = [
    #     (block_of_values,)
    #     for block_of_values in blocks
    #   ]
    #   elapsed_times = []
    #   for num_procs in [1, 2, 4, 8]:
    #     ave_elapsed_time = time_function(
    #       func        = cpu_heavy_task,
    #       args_list   = args_list,
    #       num_repeats = 5,
    #       num_procs   = num_procs,
    #       verbose     = False
    #     )
    #     elapsed_times.append(ave_elapsed_time)
    #   for pair_index in range(len(elapsed_times)-1):
    #     self.assertGreater(elapsed_times[pair_index], elapsed_times[pair_index+1])

    # def test_parallel_correctness(self):
    #   num_values_per_block = 10
    #   num_blocks = 6
    #   blocks = [
    #     [
    #       float(x)
    #       for x in range(num_values_per_block)
    #     ] for _ in range(num_blocks)
    #   ]
    #   args_list = [
    #     (block_of_values,)
    #     for block_of_values in blocks
    #   ]
    #   expected_results = [
    #     cpu_heavy_task(block_of_values)
    #     for block_of_values in blocks
    #   ]
    #   results = independent_tasks.run_in_parallel(
    #     func          = cpu_heavy_task,
    #     args_list     = args_list,
    #     num_procs     = 2,
    #     show_progress = False
    #   )
    #   self.assertEqual(len(results), len(expected_results))
    #   for result, expected in zip(results, expected_results):
    #     self.assertEqual(result, expected)

    # def test_empty_args_list(self):
    #   args_list = []
    #   result = independent_tasks.run_in_parallel(
    #       func          = dummy_task,
    #       args_list     = args_list,
    #       num_procs     = 2,
    #       show_progress = False
    #   )
    #   self.assertEqual(result, [])

    # def test_exception_propagation(self):
    #   args_list = [()] * 3
    #   with self.assertRaises(RuntimeError) as cm:
    #     independent_tasks.run_in_parallel(
    #       func          = error_task,
    #       args_list     = args_list,
    #       num_procs     = 2,
    #       show_progress = False
    #     )
    #   error_lines = str(cm.exception).split('\n')[1:]
    #   self.assertEqual(len(error_lines), 3)
    #   self.assertTrue(all("ValueError" in line for line in error_lines))

    # def test_mixed_success_failure(self):
    #   args_list = [(i,) for i in range(10)]
    #   with self.assertRaises(RuntimeError) as cm:
    #     independent_tasks.run_in_parallel(
    #       func          = mixed_task,
    #       args_list     = args_list,
    #       num_procs     = 2,
    #       show_progress = False
    #     )
    #   error = cm.exception
    #   self.assertIn("Task 5 failed", str(error))
    #   self.assertIn("Task 5 failed: ZeroDivisionError", str(error))

    # def test_process_expiry_handling(self):
    #   args_list = [()] * 3
    #   with self.assertRaises(RuntimeError) as cm:
    #     independent_tasks.run_in_parallel(
    #         func          = crashing_task,
    #         args_list     = args_list,
    #         num_procs     = 2,
    #         show_progress = False
    #     )
    #   self.assertIn("ProcessExpired", str(cm.exception))

    # def test_result_ordering(self):
    #   args_list = [(0.2, 3), (10.1, 1), (0.3, 4), (0.0, 2)]
    #   results = independent_tasks.run_in_parallel(
    #     func          = delayed_return,
    #     args_list     = args_list,
    #     num_procs     = 4,
    #     show_progress = False
    #   )
    #   self.assertEqual(results, [3, 1, 4, 2])

    # def test_various_data_types(self):
    #   args_list = [
    #     ("hello",),
    #     ({"key": "value"},),
    #     (123,),
    #     (b"bytes",),
    #   ]
    #   results = independent_tasks.run_in_parallel(
    #     func          = dummy_task,
    #     args_list     = args_list,
    #     num_procs     = 2,
    #     show_progress = False
    #   )
    #   expected_reults = [
    #     arg[0]
    #     for arg in args_list
    #   ]
    #   self.assertEqual(results, expected_reults)


if __name__ == "__main__":
    unittest.main()
