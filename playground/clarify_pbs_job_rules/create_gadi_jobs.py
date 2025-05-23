def create_pbs_job_script(
    queue_name      : str,
    compute_group   : str,
    num_procs       : int,
    wall_time_hours : int,
  ) -> None:
  file_name = f"{queue_name}_{compute_group}_{num_procs:d}cpus_{wall_time_hours:d}hrs.pbs"
  walltime = f"{wall_time_hours:02}:00:00"
  command = [
    'echo "Starting job ${PBS_JOBID}"',
    "hostname",
    'echo "Cores available: $(nproc)"',
    'echo "Job completed successfully"'
  ]
  pbs_headers = [
    "#!/bin/bash",
    f"#PBS -P {compute_group}",
    f"#PBS -q {queue_name}",
    f"#PBS -l walltime={walltime}",
    f"#PBS -l ncpus={num_procs}",
    "#PBS -j oe",
    f"#PBS -N rule_test_{num_procs}cpu",
    "#PBS -o output.log"
  ]
  with open(file_name, "w") as f:
    f.write("\n".join(pbs_headers))
    f.write("\n\n")
    f.write("\n".join(command))
    f.write("\n")
  print(f"Created: {file_name}")

def main():
  num_cpus_per_node = 48
  test_params = [
    ("normal", "jh2", 672 - num_cpus_per_node, 50),
    ("normal", "jh2", 1440 - num_cpus_per_node, 50),
    ("normal", "jh2", 2976 - num_cpus_per_node, 50),
    ("normal", "jh2", 20736 - num_cpus_per_node, 50),
  ]
  for params in test_params:
    create_pbs_job_script(*params)

if __name__ == "__main__":
  main()

## end