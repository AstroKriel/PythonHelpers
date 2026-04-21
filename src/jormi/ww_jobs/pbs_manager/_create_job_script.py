## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## local
## import directly from the module file (not via the package __init__) to avoid a static import cycle
from jormi.ww_jobs.pbs_manager import _job_validation
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
    queue_name: str,
    compute_group_name: str,
    num_procs: int,
    memory_gb: int,
    wall_time_hours: int,
    storage_group_name: str | None,
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
        param=queue_name,
        param_name="queue_name",
    )
    check_types.ensure_nonempty_string(
        param=compute_group_name,
        param_name="compute_group_name",
    )
    check_types.ensure_finite_int(
        param=num_procs,
        param_name="num_procs",
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
        param=storage_group_name,
        param_name="storage_group_name",
        allow_none=True,
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


def _build_pbs_script(
    *,
    tag_name: str,
    compute_group_name: str,
    queue_name: str,
    wall_time_string: str,
    num_procs: int,
    memory_gb: int,
    storage_group_name: str,
    email_address: str | None,
    mail_options: str,
    prep_command: str | None,
    main_command: str,
    post_command: str | None,
    always_run_post: bool,
) -> list[str]:
    lines: list[str] = []
    ## --- pbs header
    lines += [
        "#!/bin/bash",
        f"#PBS -P {compute_group_name}",
        f"#PBS -q {queue_name}",
        f"#PBS -l walltime={wall_time_string}",
        f"#PBS -l ncpus={num_procs}",
        f"#PBS -l mem={memory_gb}GB",
        f"#PBS -l storage=scratch/{storage_group_name}+gdata/{storage_group_name}",
        "#PBS -l wd",
        f"#PBS -N {tag_name}",
        "#PBS -j oe",
    ]
    if email_address is not None:
        lines += [
            f"#PBS -m {mail_options}",
            f"#PBS -M {email_address}",
        ]
    ## --- shell setup + logging
    lines += [
        "",
        "set -euo pipefail",
        f'LOG_FILE="{tag_name}.out"',
        'exec >"$LOG_FILE" 2>&1',
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


def create_pbs_job_script(
    *,
    system_name: str,
    directory: str | Path,
    file_name: str,
    main_command: str,
    tag_name: str,
    queue_name: str,
    compute_group_name: str,
    num_procs: int,
    memory_gb: int,
    wall_time_hours: int,
    storage_group_name: str | None = None,
    prep_command: str | None = None,
    post_command: str | None = None,
    always_run_post: bool = True,
    email_address: str | None = None,
    email_on_start: bool = False,
    email_on_finish: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Create a PBS job script with optional pre/post steps.

    Parameters
    ---
    - `system_name`:
        Name of the HPC system (e.g. `"gadi"`). Used to look up valid queue and
        compute group combinations.

    - `directory`:
        Directory where the job script file will be written.

    - `file_name`:
        Filename for the job script; must end with `.sh`.

    - `main_command`:
        Primary workload command. Its exit code is captured and used as the
        script's final exit code.

    - `tag_name`:
        PBS job name (`-N`). Also used as the log file name.

    - `queue_name`:
        PBS queue to submit to (e.g. `"normal"`, `"rsaa"`).

    - `compute_group_name`:
        NCI compute allocation group (e.g. `"jh2"`, `"ek9"`).

    - `num_procs`:
        Number of CPUs to request (`-l ncpus`).

    - `memory_gb`:
        Memory limit in GB (`-l mem`).

    - `wall_time_hours`:
        Maximum wall time in hours.

    - `storage_group_name`:
        NCI storage allocation group for scratch/gdata access. Defaults to
        `compute_group_name` when `None`.

    - `prep_command`:
        Optional command that runs before `main_command` (e.g. loading modules,
        activating an environment).

    - `post_command`:
        Optional command that runs after `main_command`. Runs always when
        `always_run_post` is `True`; only on success otherwise.

    - `always_run_post`:
        When `True`, `post_command` runs even if `main_command` fails.

    - `email_address`:
        If provided, PBS sends job notifications to this address. Failure
        notifications are always included.

    - `email_on_start`:
        When `True`, sends a notification when the job begins (`-m b`).

    - `email_on_finish`:
        When `True`, sends a notification when the job ends (`-m e`).

    - `verbose`:
        When `True`, prints a summary of job parameters after writing the script.
    """
    _validate_inputs(
        system_name=system_name,
        directory=directory,
        file_name=file_name,
        main_command=main_command,
        tag_name=tag_name,
        queue_name=queue_name,
        compute_group_name=compute_group_name,
        num_procs=num_procs,
        memory_gb=memory_gb,
        wall_time_hours=wall_time_hours,
        storage_group_name=storage_group_name,
        prep_command=prep_command,
        post_command=post_command,
        always_run_post=always_run_post,
        email_address=email_address,
        email_on_start=email_on_start,
        email_on_finish=email_on_finish,
        verbose=verbose,
    )
    if storage_group_name is None:
        storage_group_name = compute_group_name
    if compute_group_name != storage_group_name:
        print(
            f"Note: `compute_group_name` = {compute_group_name} and `storage_group_name` = {storage_group_name} are different.",
        )
    wall_time_string = f"{wall_time_hours:02}:00:00"
    try:
        _job_validation.validate_job_params(
            system_name,
            queue_name,
            compute_group_name,
            num_procs,
            wall_time_hours,
        )
    except _job_validation.QueueValidationError as error:
        raise ValueError(f"Invalid job parameters: {error}")
    mail_options = "a"  # notify on failure
    if email_on_start:
        mail_options += "b"
    if email_on_finish:
        mail_options += "e"
    file_path = Path(directory) / file_name
    file_path = _ensure_path_is_valid(file_path=file_path)
    lines = _build_pbs_script(
        tag_name=tag_name,
        compute_group_name=compute_group_name,
        queue_name=queue_name,
        wall_time_string=wall_time_string,
        num_procs=num_procs,
        memory_gb=memory_gb,
        storage_group_name=storage_group_name,
        email_address=email_address,
        mail_options=mail_options,
        prep_command=prep_command,
        main_command=main_command,
        post_command=post_command,
        always_run_post=always_run_post,
    )
    file_path.write_text("\n".join(lines) + "\n")
    if verbose:
        print("[Created PBS Job]")
        print(file_path)
        print(f"\t> Tagname  : {tag_name}")
        print(f"\t> CPUs     : {num_procs}")
        print(f"\t> Memory   : {memory_gb} GB")
        print(f"\t> Walltime : {wall_time_string}")
    return file_path


## } MODULE
