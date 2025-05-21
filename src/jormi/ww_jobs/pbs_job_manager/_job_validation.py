## START OF MODULE


## ###############################################################
## GLOBAL PARAMETERS
## ###############################################################

_QUEUE_CONFIGS = {
  "gadi" : {
    "normal": {
      "cost_per_cpu_hour": 2,
      "cpus_per_node": 48,
      "max_walltime": 48,
      "max_cpus": 20736,
      "walltime_limits": [
        {"threshold_cpus" :   672, "max_hours" : 48},
        {"threshold_cpus" :  1440, "max_hours" : 24},
        {"threshold_cpus" :  2976, "max_hours" : 10},
        {"threshold_cpus" : 20736, "max_hours" :  5},
      ],
    },
    "rsaa": {
      "cost_per_cpu_hour" : 1,
      "cpus_per_node": 104,
      "max_walltime": 48,
      "max_cpus": 1248,
    },
  }
}


## ###############################################################
## CUSTOM ERROR MESSAGE
## ###############################################################

class QueueValidationError(ValueError):
  """Exception raised when job parameters do not meet queue constraints."""
  pass


## ###############################################################
## FUNCTIONS
## ###############################################################

def validate_job_params(
    system_name     : str,
    queue_name      : str,
    num_procs       : int,
    wall_time_hours : int,
  ) -> None:
  queue_config = _get_queue_config(system_name, queue_name)
  _validate_cpu_alignment(queue_name, num_procs, queue_config)
  _validate_cpu_limit(queue_name, num_procs, queue_config)
  _validate_walltime(queue_name, num_procs, wall_time_hours, queue_config)

def _get_queue_config(
    system_name : str,
    queue_name  : str
  ) -> dict:
  queue_config = _QUEUE_CONFIGS.get(system_name).get(queue_name)
  if not queue_config:
    raise QueueValidationError(f"Unknown queue `{queue_name}` for system `{system_name}`.")
  return queue_config

def _validate_cpu_alignment(
    queue_name   : str,
    num_procs    : int,
    queue_config : dict
  ) -> None:
  cpus_per_node = queue_config.get("cpus_per_node")
  if cpus_per_node is None:
    raise QueueValidationError(f"No `cpus_per_node` specified for queue `{queue_name}`.")
  if (num_procs > cpus_per_node) and (num_procs % cpus_per_node != 0):
    raise QueueValidationError(
      f"Queue `{queue_name}` requires CPU requests above one node to be in multiples of {cpus_per_node}. "
      f"Requested: {num_procs}."
    )

def _validate_cpu_limit(
    queue_name   : str,
    num_procs    : int,
    queue_config : dict
  ) -> None:
  max_cpus = queue_config.get("max_cpus")
  if (max_cpus is not None) and (num_procs > max_cpus):
    raise QueueValidationError(f"Queue `{queue_name}` allows a maximum of {max_cpus} CPUs. Requested: {num_procs}.")

def _validate_walltime(
    queue_name      : str,
    num_procs       : int,
    wall_time_hours : int,
    queue_config    : dict
  ) -> None:
  walltime_limits = queue_config.get("walltime_limits")
  if walltime_limits:
    sorted_rules = sorted(
      walltime_limits,
      key = lambda rule: rule["threshold_cpus"]
    )
    matching_rule = next(
      (
        rule
        for rule in sorted_rules
        if num_procs <= rule["threshold_cpus"]
      ),
      default = None
    )
    if matching_rule is None:
      raise QueueValidationError(f"CPU request ({num_procs}) exceeds all walltime rules in `{queue_name}`.")
    if wall_time_hours > matching_rule["max_hours"]:
      raise QueueValidationError(
        f"Max walltime is {matching_rule['max_hours']}h for ≤{matching_rule['threshold_cpus']} CPUs "
        f"in `{queue_name}` queue."
      )
    return
  max_walltime = queue_config.get("max_walltime")
  if max_walltime is not None and wall_time_hours > max_walltime:
    raise QueueValidationError(
      f"Max walltime is {max_walltime}h in `{queue_name}` queue. "
      f"Requested: {wall_time_hours}h."
    )


## END OF MODULE