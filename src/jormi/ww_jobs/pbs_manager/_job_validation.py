## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

##
## === GLOBAL PARAMS
##

_QUEUE_CONFIGS: dict[str, dict[str, Any]] = {
    "gadi": {
        "normal": {
            "cost_per_cpu_hour": 2,
            "cpus_per_node": 48,
            "max_wall_time": 48,
            "max_cpus": 20736,
            "wall_time_limits": [
                {
                    "threshold_cpus": 672,
                    "max_hours": 48,
                },
                {
                    "threshold_cpus": 1440,
                    "max_hours": 24,
                },
                {
                    "threshold_cpus": 2976,
                    "max_hours": 10,
                },
                {
                    "threshold_cpus": 20736,
                    "max_hours": 5,
                },
            ],
        },
        "rsaa": {
            "cost_per_cpu_hour": 1,
            "cpus_per_node": 104,
            "max_wall_time": 48,
            "max_cpus": 1248,
        },
    },
}

_COMPUTE_GROUP_CONFIGS: dict[str, dict[str, set[str]]] = {
    "gadi": {
        "jh2": {"normal", "rsaa"},
        "ek9": {"normal", "rsaa"},
        "mk27": {"rsaa"},
    },
}

##
## === CUSTOM ERROR
##


class QueueValidationError(ValueError):
    """Exception raised when job parameters do not meet queue constraints."""

    pass


##
## === FUNCTIONS
##


def validate_job_params(
    system_name: str,
    queue_name: str,
    compute_group_name: str,
    num_procs: int,
    wall_time_hours: int,
) -> None:
    """Confirm that the requested job parameters meet the system-queue constraints."""
    queue_config = _get_queue_config(
        system_name=system_name,
        queue_name=queue_name,
    )
    _validate_queue_group_combination(
        system_name=system_name,
        queue_name=queue_name,
        compute_group_name=compute_group_name,
    )
    _validate_cpu_rules(
        queue_name=queue_name,
        num_procs=num_procs,
        queue_config=queue_config,
    )
    _validate_cpu_limit(
        queue_name=queue_name,
        num_procs=num_procs,
        queue_config=queue_config,
    )
    _validate_wall_time_rules(
        queue_name=queue_name,
        num_procs=num_procs,
        wall_time_hours=wall_time_hours,
        queue_config=queue_config,
    )


def _get_queue_config(
    system_name: str,
    queue_name: str,
) -> dict[str, Any]:
    """Extract relevant system-queue constraints."""
    system_config = _QUEUE_CONFIGS.get(system_name)
    if system_config is None:
        raise QueueValidationError(f"Unknown system `{system_name}`.")
    queue_config = system_config.get(queue_name)
    if queue_config is None:
        raise QueueValidationError(
            f"Unknown queue `{queue_name}` for system `{system_name}`.",
        )
    return queue_config


def _validate_queue_group_combination(
    system_name: str,
    queue_name: str,
    compute_group_name: str,
) -> None:
    system_config = _COMPUTE_GROUP_CONFIGS.get(system_name)
    if system_config is None:
        raise QueueValidationError(f"Unknown system `{system_name}`.")
    valid_queues = system_config.get(compute_group_name)
    if valid_queues is None:
        raise QueueValidationError(
            f"Unknown compute group `{compute_group_name}` for system `{system_name}`.",
        )
    if queue_name not in valid_queues:
        raise QueueValidationError(
            f"Queue `{queue_name}` is not valid for compute group `{compute_group_name}` "
            f"on system `{system_name}`. Valid queues: {sorted(valid_queues)}.",
        )


def _validate_cpu_rules(
    queue_name: str,
    num_procs: int,
    queue_config: dict[str, Any],
) -> None:
    cpus_per_node = queue_config.get("cpus_per_node")
    if cpus_per_node is None:
        raise QueueValidationError(f"No `cpus_per_node` specified for queue `{queue_name}`.")
    ## request all node-cpus if more than one node is used
    if (num_procs > cpus_per_node) and (num_procs % cpus_per_node != 0):
        raise QueueValidationError(
            f"Queue `{queue_name}` requires CPU requests above one node to be in multiples of {cpus_per_node}. "
            f"Requested: {num_procs}.",
        )


def _validate_cpu_limit(
    queue_name: str,
    num_procs: int,
    queue_config: dict[str, Any],
) -> None:
    max_cpus = queue_config.get("max_cpus")
    if (max_cpus is not None) and (num_procs > max_cpus):
        raise QueueValidationError(
            f"Queue `{queue_name}` allows a maximum of {max_cpus} CPUs. Requested: {num_procs}.",
        )


def _validate_wall_time_rules(
    queue_name: str,
    num_procs: int,
    wall_time_hours: int,
    queue_config: dict[str, Any],
) -> None:
    wall_time_limits = queue_config.get("wall_time_limits")
    if wall_time_limits:
        ## sort wall time rules by threshold cpu-count (ascending)
        sorted_rules = sorted(
            wall_time_limits,
            key=lambda rule: rule["threshold_cpus"],
        )
        ## find the first rule which is met
        matching_rule = next(
            (rule for rule in sorted_rules if num_procs <= rule["threshold_cpus"]),
            None,
        )
        if matching_rule is None:
            raise QueueValidationError(
                f"CPU request ({num_procs}) exceeds all wall_time rules in `{queue_name}`.",
            )
        if wall_time_hours > matching_rule["max_hours"]:
            raise QueueValidationError(
                f"Max wall_time is {matching_rule['max_hours']}h for ≤{matching_rule['threshold_cpus']} CPUs "
                f"in `{queue_name}` queue.",
            )
        return
    ## check that the job does not exceed the maximum wall_time
    max_wall_time = queue_config.get("max_wall_time")
    if (max_wall_time is not None) and wall_time_hours > max_wall_time:
        raise QueueValidationError(
            f"Max wall time is {max_wall_time}h in `{queue_name}` queue. "
            f"Requested: {wall_time_hours}h.",
        )


## } MODULE
