## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## local
from jormi.ww_io import manage_shell

##
## === FUNCTIONS
##


def submit_job(
    *,
    directory: str | Path,
    file_name: str,
    check_status: bool = False,
) -> bool:
    """Submit a SLURM job script via `sbatch`."""
    directory = Path(directory).resolve()
    if check_status and is_job_already_in_queue(
        directory=directory,
        file_name=file_name,
    ):
        print("Job is already currently running:", file_name)
        return False
    print("Submitting job:", file_name)
    try:
        manage_shell.execute_shell_command(
            command=f"sbatch {file_name}",
            working_directory=directory,
        )
        return True
    except RuntimeError as error:
        print(f"Failed to submit job `{file_name}`: {error}")
        return False


def is_job_already_in_queue(
    *,
    directory: str | Path,
    file_name: str,
) -> bool:
    """Check if a job with the same name is already in the SLURM queue."""
    file_path = Path(directory) / file_name
    if not file_path.is_file():
        print(f"`{file_name}` job file does not exist in: {directory}")
        return False
    job_tag = get_job_tag_from_slurm_script(file_path=file_path)
    if not job_tag:
        print(f"`#SBATCH --job-name` not found in job file: {file_name}")
        return False
    queued_jobs = get_list_of_queued_jobs()
    if not queued_jobs:
        return False
    queued_job_tags = [tag for _, tag in queued_jobs]
    return job_tag in queued_job_tags


def get_job_tag_from_slurm_script(
    *,
    file_path: str | Path,
) -> str | None:
    """Extract the job name from a SLURM job script."""
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as file_pointer:
        for line in file_pointer:
            stripped = line.strip()
            if stripped.startswith("#SBATCH --job-name="):
                return stripped.split("=", 1)[-1]
    return None


def get_list_of_queued_jobs() -> list[tuple[str, str]] | None:
    """Collect all (job_id, job_name) pairs currently in the SLURM queue."""
    try:
        result = manage_shell.execute_shell_command(
            command="squeue --me --format=%i,%j --noheader",
            timeout_seconds=60,
            capture_output=True,
        )
        if not result.stdout:
            return []
        jobs: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                jobs.append((parts[0], parts[1]))
        return jobs
    except RuntimeError as error:
        print(f"Error retrieving job info from the queue: {error}")
        return None


## } MODULE
