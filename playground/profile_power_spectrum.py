import sys
import numpy

from jormi.ww_fields import compute_spectra

import time
import psutil
import cProfile
import pstats
from memory_profiler import memory_usage


def measure_execution_time(func, field, repeats=3):
    execution_times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        func(field)
        end_time = time.perf_counter()
        execution_times.append(end_time - start_time)
    avg_time = numpy.median(execution_times)
    std_time = numpy.std(execution_times)
    print(f"\t> execution took {avg_time:.5f} +/- {std_time:.5f} seconds on average.")


def measure_cpu_usage(func, field, repeats=3):
    cpu_usages = []
    for _ in range(repeats):
        process = psutil.Process()
        start_cpu = process.cpu_percent(interval=None)
        func(field)
        end_cpu = process.cpu_percent(interval=None)
        cpu_usages.append(end_cpu - start_cpu)
    avg_cpu = numpy.median(cpu_usages)
    std_cpu = numpy.std(cpu_usages)
    print(f"\t> used {avg_cpu:.2f} +/- {std_cpu:.2f}% of CPU on average.")


def measure_memory_usage(func, field, repeats=3):

    def func_eval():
        return func(field)

    memory_usages = []
    for _ in range(repeats):
        usage = memory_usage(func_eval)
        memory_usages.extend(usage)
    avg_memory = numpy.median(memory_usages)
    std_memory = numpy.std(memory_usages)
    print(f"\t> used {avg_memory:.2f} +/- {std_memory:.2f} MB on average.")


def measure_io_time(func, field, repeats=3):
    io_times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        func(field)
        end_time = time.perf_counter()
        io_times.append(end_time - start_time)
    avg_time = numpy.median(io_times)
    std_time = numpy.std(io_times)
    print(f"\t> I/O took {avg_time:.5f} +/- {std_time:.5f} seconds on average.")


def profile_function(func, field):
    print(f"\t> Using cProfile:\n")
    profiler = cProfile.Profile()
    profiler.enable()
    func(field)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("time")
    stats.print_stats(10)


def profile(func, field, repeats=3):
    print(f"Profiling using a domain with size {field.shape}...")
    measure_execution_time(func, field, repeats=repeats)
    measure_cpu_usage(func, field, repeats=repeats)
    measure_memory_usage(func, field, repeats=repeats)
    measure_io_time(func, field, repeats=repeats)
    profile_function(func, field)


def main():
    ## todo: generate field
    profile(compute_spectra.compute_1d_power_spectrum, field, repeats=3)


if __name__ == "__main__":
    main()
    sys.exit(0)

## end of script
