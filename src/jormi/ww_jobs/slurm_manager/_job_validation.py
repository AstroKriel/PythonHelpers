## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

##
## === GLOBAL PARAMS
##

## placeholder: confirm partition names, CPU limits, and wall-time limits via `sinfo` on CITA
_QUEUE_CONFIGS: dict[str, dict[str, Any]] = {
    "cita": {
        "all": {
            "max_wall_time": 72,
            "max_cpus": None,
        },
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
    *,
    system_name: str,
    partition_name: str,
    num_cpus: int,
    wall_time_hours: int,
) -> None:
    """Confirm that the requested job parameters meet the system-partition constraints."""
    queue_config = _get_queue_config(
        system_name=system_name,
        partition_name=partition_name,
    )
    _validate_cpu_limit(
        partition_name=partition_name,
        num_cpus=num_cpus,
        queue_config=queue_config,
    )
    _validate_wall_time(
        partition_name=partition_name,
        wall_time_hours=wall_time_hours,
        queue_config=queue_config,
    )


def _get_queue_config(
    *,
    system_name: str,
    partition_name: str,
) -> dict[str, Any]:
    system_config = _QUEUE_CONFIGS.get(system_name)
    if system_config is None:
        raise QueueValidationError(f"Unknown system `{system_name}`.")
    queue_config = system_config.get(partition_name)
    if queue_config is None:
        raise QueueValidationError(
            f"Unknown partition `{partition_name}` for system `{system_name}`.",
        )
    return queue_config


def _validate_cpu_limit(
    *,
    partition_name: str,
    num_cpus: int,
    queue_config: dict[str, Any],
) -> None:
    max_cpus = queue_config.get("max_cpus")
    if (max_cpus is not None) and (num_cpus > max_cpus):
        raise QueueValidationError(
            f"Partition `{partition_name}` allows a maximum of {max_cpus} CPUs. "
            f"Requested: {num_cpus}.",
        )


def _validate_wall_time(
    *,
    partition_name: str,
    wall_time_hours: int,
    queue_config: dict[str, Any],
) -> None:
    max_wall_time = queue_config.get("max_wall_time")
    if (max_wall_time is not None) and (wall_time_hours > max_wall_time):
        raise QueueValidationError(
            f"Partition `{partition_name}` allows a maximum wall time of {max_wall_time}h. "
            f"Requested: {wall_time_hours}h.",
        )


## } MODULE
