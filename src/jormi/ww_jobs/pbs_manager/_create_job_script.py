## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## local
from jormi.ww_io import manage_log
from jormi.ww_checks import check_python_types

##
## === FUNCTIONS
##


def _validate_inputs(
    *,
    directory: str | Path,
    file_name: str,
    directives: list[str] | None,
    main_command: str,
    tag_name: str,
    queue_name: str | None,
    num_procs: int | None,
    memory_gb: int | None,
    wall_time_hours: int | None,
    prep_command: str | None,
    post_command: str | None,
    always_run_post: bool,
    email_address: str | None,
    email_on_start: bool,
    email_on_finish: bool,
    verbose: bool,
) -> None:
    check_python_types.ensure_type(
        param=directory,
        valid_types=(str, Path),
        param_name="directory",
    )
    check_python_types.ensure_nonempty_string(
        param=file_name,
        param_name="file_name",
    )
    check_python_types.ensure_type(
        param=directives,
        valid_types=(list,),
        param_name="directives",
        allow_none=True,
    )
    if directives is not None:
        if len(directives) == 0:
            raise ValueError("`directives` must contain at least one PBS header line.")
        for directive in directives:
            check_python_types.ensure_nonempty_string(
                param=directive,
                param_name="directives[]",
            )
    check_python_types.ensure_nonempty_string(
        param=main_command,
        param_name="main_command",
    )
    check_python_types.ensure_nonempty_string(
        param=tag_name,
        param_name="tag_name",
    )
    check_python_types.ensure_string(
        param=queue_name,
        param_name="queue_name",
        allow_none=True,
    )
    check_python_types.ensure_finite_int(
        param=num_procs,
        param_name="num_procs",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    check_python_types.ensure_finite_int(
        param=memory_gb,
        param_name="memory_gb",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    check_python_types.ensure_finite_int(
        param=wall_time_hours,
        param_name="wall_time_hours",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    if directives is None:
        if queue_name is None:
            raise ValueError(
                "`queue_name` is required when `directives` are not provided."
            )
        if num_procs is None:
            raise ValueError(
                "`num_procs` is required when `directives` are not provided."
            )
        if wall_time_hours is None:
            raise ValueError(
                "`wall_time_hours` is required when `directives` are not provided."
            )
    check_python_types.ensure_string(
        param=prep_command,
        param_name="prep_command",
        allow_none=True,
    )
    check_python_types.ensure_string(
        param=post_command,
        param_name="post_command",
        allow_none=True,
    )
    check_python_types.ensure_bool(
        param=always_run_post,
        param_name="always_run_post",
    )
    check_python_types.ensure_string(
        param=email_address,
        param_name="email_address",
        allow_none=True,
    )
    check_python_types.ensure_bool(
        param=email_on_start,
        param_name="email_on_start",
    )
    check_python_types.ensure_bool(
        param=email_on_finish,
        param_name="email_on_finish",
    )
    check_python_types.ensure_bool(
        param=verbose,
        param_name="verbose",
    )


def _ensure_path_is_valid(
    *,
    file_path: Path,
) -> Path:
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".sh":
        raise ValueError(f"`file_path` must end with a `.sh` extension: {file_path}.")
    return file_path


def _build_pbs_script(
    *,
    directives: list[str] | None,
    tag_name: str,
    queue_name: str | None,
    num_procs: int | None,
    memory_gb: int | None,
    wall_time_hours: int | None,
    email_address: str | None,
    mail_options: str,
    prep_command: str | None,
    main_command: str,
    post_command: str | None,
    always_run_post: bool,
) -> list[str]:
    file_lines: list[str] = []
    ## --- pbs header
    if directives is None:
        assert queue_name is not None
        assert num_procs is not None
        assert wall_time_hours is not None
        directives = [
            "#!/bin/bash",
            f"#PBS -q {queue_name}",
            f"#PBS -l walltime={wall_time_hours:02}:00:00",
            f"#PBS -l ncpus={num_procs}",
            f"#PBS -N {tag_name}",
            "#PBS -j oe",
        ]
        if memory_gb is not None:
            directives.insert(4, f"#PBS -l mem={memory_gb}GB")
    file_lines += directives
    if email_address is not None:
        file_lines += [
            f"#PBS -m {mail_options}",
            f"#PBS -M {email_address}",
        ]
    ## --- shell setup + logging
    file_lines += [
        "",
        "set -euo pipefail",
        f'LOG_FILE="{tag_name}.out"',
        'exec >"$LOG_FILE" 2>&1',
    ]
    ## --- pre-command (if provided)
    if prep_command:
        file_lines += [
            "",
            "## preparation step(s)",
            prep_command.rstrip(),
        ]
    ## --- main command
    file_lines += [
        "",
        "## main workload (capture exit code)",
        "main_command_exit_code=0",
        f"{main_command.rstrip()} || main_command_exit_code=$?",
        'echo "Main command exit code: $main_command_exit_code"',
    ]
    ## --- post-command (if provided)
    if post_command:
        file_lines += [
            "",
            "## post-processing step(s)",
            "post_command_exit_code=not_run",
        ]
        if always_run_post:
            file_lines.append(f"{post_command.rstrip()} || post_command_exit_code=$?")
        else:
            file_lines += [
                'if [ "$main_command_exit_code" -eq 0 ]; then',
                f"\t{post_command.rstrip()} || post_command_exit_code=$?",
                "fi",
            ]
        file_lines.append('echo "Post command exit code: $post_command_exit_code"')
    ## --- exit with the main command's exit code
    file_lines += [
        "",
        "## exit with the main command's exit code",
        'exit "$main_command_exit_code"',
    ]
    return file_lines


def create_pbs_job_script(
    *,
    directory: str | Path,
    file_name: str,
    directives: list[str] | None = None,
    main_command: str,
    tag_name: str,
    queue_name: str | None = None,
    num_procs: int | None = None,
    memory_gb: int | None = None,
    wall_time_hours: int | None = None,
    prep_command: str | None = None,
    post_command: str | None = None,
    always_run_post: bool = True,
    email_address: str | None = None,
    email_on_start: bool = False,
    email_on_finish: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Create a PBS job script with caller-provided directives or generic resources.

    Parameters
    ---
    - `directory`:
        Directory where the job script file will be written.

    - `file_name`:
        Filename for the job script; must end with `.sh`.

    - `directives`:
        Ordered PBS header lines to write at the top of the script. This should
        include the shebang and any `#PBS ...` directives required by the
        target system. When omitted, a generic PBS header is built from
        `queue_name`, `num_procs`, `memory_gb`, and `wall_time_hours`.

    - `main_command`:
        Primary workload command. Its exit code is captured and used as the
        script's final exit code.

    - `tag_name`:
        Job tag used for the local log file name.

    - `queue_name`, `num_procs`, `memory_gb`, `wall_time_hours`:
        Generic PBS resource settings used when `directives` are not supplied.

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
        directory=directory,
        file_name=file_name,
        directives=directives,
        main_command=main_command,
        tag_name=tag_name,
        queue_name=queue_name,
        num_procs=num_procs,
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
    mail_options = "a"  # notify on failure
    if email_on_start:
        mail_options += "b"
    if email_on_finish:
        mail_options += "e"
    file_path = Path(directory) / file_name
    file_path = _ensure_path_is_valid(file_path=file_path)
    file_lines = _build_pbs_script(
        directives=directives,
        tag_name=tag_name,
        queue_name=queue_name,
        num_procs=num_procs,
        memory_gb=memory_gb,
        wall_time_hours=wall_time_hours,
        email_address=email_address,
        mail_options=mail_options,
        prep_command=prep_command,
        main_command=main_command,
        post_command=post_command,
        always_run_post=always_run_post,
    )
    file_path.write_text("\n".join(file_lines) + "\n")
    if verbose:
        manage_log.log_action(
            title="Create PBS job",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="Wrote job script.",
            notes={
                "file": str(file_path),
                "tag_name": tag_name,
            },
        )
    return file_path


## } MODULE
