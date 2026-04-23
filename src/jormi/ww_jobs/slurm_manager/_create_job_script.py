## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## local
from jormi.ww_io import manage_log
## import directly from the module file (not via the package __init__) to avoid a static import cycle
from jormi.ww_jobs.slurm_manager import _job_validation
from jormi.ww_types import check_types

##
## === FUNCTIONS
##


def _validate_inputs(
    *,
    system_name: str,
    directory: str | Path,
    file_name: str,
    main_command: str,
    tag_name: str,
    partition_name: str,
    num_cpus: int,
    memory_gb: int,
    wall_time_hours: int,
    prep_command: str | None,
    post_command: str | None,
    always_run_post: bool,
    email_address: str | None,
    email_on_start: bool,
    email_on_finish: bool,
    verbose: bool,
) -> None:
    check_types.ensure_nonempty_string(
        param=system_name,
        param_name="system_name",
    )
    check_types.ensure_type(
        param=directory,
        valid_types=(str, Path),
        param_name="directory",
    )
    check_types.ensure_nonempty_string(
        param=file_name,
        param_name="file_name",
    )
    check_types.ensure_nonempty_string(
        param=main_command,
        param_name="main_command",
    )
    check_types.ensure_nonempty_string(
        param=tag_name,
        param_name="tag_name",
    )
    check_types.ensure_nonempty_string(
        param=partition_name,
        param_name="partition_name",
    )
    check_types.ensure_finite_int(
        param=num_cpus,
        param_name="num_cpus",
        require_positive=True,
        allow_zero=False,
    )
    check_types.ensure_finite_int(
        param=memory_gb,
        param_name="memory_gb",
        require_positive=True,
        allow_zero=False,
    )
    check_types.ensure_finite_int(
        param=wall_time_hours,
        param_name="wall_time_hours",
        require_positive=True,
        allow_zero=False,
    )
    check_types.ensure_string(
        param=prep_command,
        param_name="prep_command",
        allow_none=True,
    )
    check_types.ensure_string(
        param=post_command,
        param_name="post_command",
        allow_none=True,
    )
    check_types.ensure_bool(
        param=always_run_post,
        param_name="always_run_post",
    )
    check_types.ensure_string(
        param=email_address,
        param_name="email_address",
        allow_none=True,
    )
    check_types.ensure_bool(
        param=email_on_start,
        param_name="email_on_start",
    )
    check_types.ensure_bool(
        param=email_on_finish,
        param_name="email_on_finish",
    )
    check_types.ensure_bool(
        param=verbose,
        param_name="verbose",
    )


def _ensure_path_is_valid(
    *,
    file_path: Path,
) -> Path:
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".sh":
        raise ValueError(f"`file_path` must end with a .sh extension: {file_path}")
    return file_path


def _build_slurm_script(
    *,
    tag_name: str,
    partition_name: str,
    num_cpus: int,
    memory_gb: int,
    wall_time_string: str,
    email_address: str | None,
    email_events_string: str,
    prep_command: str | None,
    main_command: str,
    post_command: str | None,
    always_run_post: bool,
) -> list[str]:
    lines: list[str] = []
    ## --- slurm header
    lines += [
        "#!/bin/bash",
        f"#SBATCH --job-name={tag_name}",
        f"#SBATCH --partition={partition_name}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={num_cpus}",
        f"#SBATCH --mem={memory_gb}G",
        f"#SBATCH --time={wall_time_string}",
        "#SBATCH --output=%x_%j.out",
        "#SBATCH --error=%x_%j.err",
    ]
    if email_address is not None:
        lines += [
            f"#SBATCH --mail-user={email_address}",
            f"#SBATCH --mail-type={email_events_string}",
        ]
    ## --- shell setup
    lines += [
        "",
        "set -euo pipefail",
    ]
    ## --- pre-command (if provided)
    if prep_command:
        lines += [
            "",
            "## preparation step(s)",
            prep_command.rstrip(),
        ]
    ## --- main command
    lines += [
        "",
        "## main workload (capture exit code)",
        "main_command_exit_code=0",
        f"{main_command.rstrip()} || main_command_exit_code=$?",
        'echo "Main command exit code: $main_command_exit_code"',
    ]
    ## --- post-command (if provided)
    if post_command:
        lines += ["", "## post-processing step(s)", "post_command_exit_code=not_run"]
        if always_run_post:
            lines.append(f"{post_command.rstrip()} || post_command_exit_code=$?")
        else:
            lines += [
                'if [ "$main_command_exit_code" -eq 0 ]; then',
                f"\t{post_command.rstrip()} || post_command_exit_code=$?",
                "fi",
            ]
        lines.append('echo "Post command exit code: $post_command_exit_code"')
    ## --- exit with the main command's exit code
    lines += [
        "",
        "## exit with the main command's exit code",
        'exit "$main_command_exit_code"',
    ]
    return lines


def create_slurm_job_script(
    *,
    system_name: str,
    directory: str | Path,
    file_name: str,
    main_command: str,
    tag_name: str,
    partition_name: str,
    num_cpus: int,
    memory_gb: int,
    wall_time_hours: int,
    prep_command: str | None = None,
    post_command: str | None = None,
    always_run_post: bool = True,
    email_address: str | None = None,
    email_on_start: bool = False,
    email_on_finish: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Create a SLURM job script with optional pre/post steps.

    Parameters
    ---
    - `system_name`:
        Name of the HPC system. Used to look up partition constraints.

    - `directory`:
        Directory where the job script file will be written.

    - `file_name`:
        Filename for the job script; must end with `.sh`.

    - `main_command`:
        Primary workload command. Its exit code is captured and used as the
        script's final exit code.

    - `prep_command`:
        Optional command that runs before `main_command` (e.g. loading modules,
        activating an environment).

    - `post_command`:
        Optional command that runs after `main_command`. Runs always when
        `always_run_post` is `True`; only on success otherwise.

    - `always_run_post`:
        When `True`, `post_command` will run even if `main_command` fails.

    - `tag_name`:
        SLURM job name (`--job-name`). Also used as the prefix for log files.

    - `partition_name`:
        SLURM partition to submit the job to.

    - `num_cpus`:
        Number of CPUs to request (`--cpus-per-task`).

    - `memory_gb`:
        Memory limit in GB.

    - `wall_time_hours`:
        Maximum wall time in hours.

    - `email_address`:
        If provided, SLURM sends job notifications to this address. Failure
        notifications are always included.

    - `email_on_start`:
        When `True`, sends a notification when the job begins.

    - `email_on_finish`:
        When `True`, sends a notification when the job ends.

    - `verbose`:
        When `True`, prints a summary of job parameters after writing the script.
    """
    _validate_inputs(
        system_name=system_name,
        directory=directory,
        file_name=file_name,
        main_command=main_command,
        tag_name=tag_name,
        partition_name=partition_name,
        num_cpus=num_cpus,
        memory_gb=memory_gb,
        wall_time_hours=wall_time_hours,
        prep_command=prep_command,
        post_command=post_command,
        always_run_post=always_run_post,
        email_address=email_address,
        email_on_start=email_on_start,
        email_on_finish=email_on_finish,
        verbose=verbose,
    )
    try:
        _job_validation.validate_job_params(
            system_name=system_name,
            partition_name=partition_name,
            num_cpus=num_cpus,
            wall_time_hours=wall_time_hours,
        )
    except _job_validation.QueueValidationError as error:
        raise ValueError(f"Invalid job parameters: {error}")
    wall_time_string = f"{wall_time_hours:02}:00:00"
    email_events: list[str] = ["FAIL"]
    if email_on_start:
        email_events.append("BEGIN")
    if email_on_finish:
        email_events.append("END")
    email_events_string = ",".join(email_events)
    file_path = Path(directory) / file_name
    file_path = _ensure_path_is_valid(file_path=file_path)
    lines = _build_slurm_script(
        tag_name=tag_name,
        partition_name=partition_name,
        num_cpus=num_cpus,
        memory_gb=memory_gb,
        wall_time_string=wall_time_string,
        email_address=email_address,
        email_events_string=email_events_string,
        prep_command=prep_command,
        main_command=main_command,
        post_command=post_command,
        always_run_post=always_run_post,
    )
    file_path.write_text("\n".join(lines) + "\n")
    if verbose:
        manage_log.log_action(
            title="Create SLURM job",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="Wrote job script.",
            notes={
                "file": str(file_path),
                "tag_name": tag_name,
                "partition": partition_name,
                "cpus": num_cpus,
                "memory": f"{memory_gb} GB",
                "walltime": wall_time_string,
            },
        )
    return file_path


## } MODULE
