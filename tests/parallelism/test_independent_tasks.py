import time
import numpy
import unittest
from jormi.parallelism import independent_tasks

def dummy_task(values):
  return sum(values)


def cpu_heavy_task(block_of_values):
  total = 0.0
  for _ in range(200):
    for value in block_of_values:
      total += value ** 1.00001
  return total

def time_function(func, args_list, num_repeats, num_procs, verbose=True):
  elapsed_times = []
  for _ in range(num_repeats):
    start_time = time.perf_counter()
    independent_tasks.run_in_parallel(
      func          = func,
      args_list     = args_list,
      num_procs     = num_procs,
      show_progress = False
    )
    elapsed_times.append(time.perf_counter() - start_time)
  ave_elapsed_time = numpy.median(elapsed_times)
  std_elapsed_time = numpy.std(elapsed_times)
  if verbose: print(f"{num_procs:d} procs completed in {ave_elapsed_time:.3f} ± {std_elapsed_time:.3f} seconds.")
  return ave_elapsed_time

def sleepy_task(duration):
  time.sleep(duration)


class TestParallelExecution(unittest.TestCase):

  def test_timeout(self):
    args_list = [(d,) for d in [0.5, 1, 3, 5]]
    try:
      independent_tasks.run_in_parallel(
        func            = sleepy_task,
        args_list       = args_list,
        timeout_seconds = 1.5,
        num_procs       = 2,
        show_progress   = False
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
    blocks = [
      [
        float(x)
        for x in range(num_values_per_block)
      ] for _ in range(num_blocks)
    ]
    args_list = [
      (block_of_values,)
      for block_of_values in blocks
    ]
    elapsed_times = []
    for num_procs in [1, 2, 4, 8]:
      ave_elapsed_time = time_function(
        func        = cpu_heavy_task,
        args_list   = args_list,
        num_repeats = 5,
        num_procs   = num_procs,
        verbose     = False
      )
      elapsed_times.append(ave_elapsed_time)
    for pair_index in range(len(elapsed_times)-1):
      self.assertGreater(elapsed_times[pair_index], elapsed_times[pair_index+1])

  def test_parallel_correctness(self):
    num_values_per_block = 10
    num_blocks = 6
    blocks = [
      [
        float(x)
        for x in range(num_values_per_block)
      ] for _ in range(num_blocks)
    ]
    args_list = [
      (block_of_values,)
      for block_of_values in blocks
    ]
    expected_results = [
      cpu_heavy_task(block_of_values)
      for block_of_values in blocks
    ]
    results = independent_tasks.run_in_parallel(
      func          = cpu_heavy_task,
      args_list     = args_list,
      num_procs     = 2,
      show_progress = False
    )
    self.assertEqual(len(results), len(expected_results))
    for result, expected in zip(results, expected_results):
      self.assertEqual(result, expected)

  def test_empty_args_list(self):
    args_list = []
    result = independent_tasks.run_in_parallel(
        func          = dummy_task,
        args_list     = args_list,
        num_procs     = 2,
        show_progress = False
    )
    self.assertEqual(result, [])


if __name__ == "__main__":
  unittest.main()
