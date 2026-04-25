## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## local
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_python_types

##
## === FUNCTIONS
##


def _ensure_inputs(
    *,
    directory: str | Path,
    file_name: str,
    directives: list[str] | None,
    main_command: str,
    tag_name: str,
    partition_name: str | None,
    num_cpus: int | None,
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
    validate_python_types.ensure_type(
        param=directory,
        valid_types=(str, Path),
        param_name="directory",
    )
    validate_python_types.ensure_nonempty_string(
        param=file_name,
        param_name="file_name",
    )
    validate_python_types.ensure_type(
        param=directives,
        valid_types=(list,),
        param_name="directives",
        allow_none=True,
    )
    if directives is not None:
        if len(directives) == 0:
            raise ValueError("`directives` must contain at least one SLURM header line.")
        for directive in directives:
            validate_python_types.ensure_nonempty_string(
                param=directive,
                param_name="directives[]",
            )
    validate_python_types.ensure_nonempty_string(
        param=main_command,
        param_name="main_command",
    )
    validate_python_types.ensure_nonempty_string(
        param=tag_name,
        param_name="tag_name",
    )
    validate_python_types.ensure_string(
        param=partition_name,
        param_name="partition_name",
        allow_none=True,
    )
    validate_python_types.ensure_finite_int(
        param=num_cpus,
        param_name="num_cpus",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    validate_python_types.ensure_finite_int(
        param=memory_gb,
        param_name="memory_gb",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    validate_python_types.ensure_finite_int(
        param=wall_time_hours,
        param_name="wall_time_hours",
        allow_none=True,
        require_positive=True,
        allow_zero=False,
    )
    if directives is None:
        if partition_name is None:
            raise ValueError("`partition_name` is required when `directives` are not provided.")
        if num_cpus is None:
            raise ValueError("`num_cpus` is required when `directives` are not provided.")
        if wall_time_hours is None:
            raise ValueError("`wall_time_hours` is required when `directives` are not provided.")
    validate_python_types.ensure_string(
        param=prep_command,
        param_name="prep_command",
        allow_none=True,
    )
    validate_python_types.ensure_string(
        param=post_command,
        param_name="post_command",
        allow_none=True,
    )
    validate_python_types.ensure_bool(
        param=always_run_post,
        param_name="always_run_post",
    )
    validate_python_types.ensure_string(
        param=email_address,
        param_name="email_address",
        allow_none=True,
    )
    validate_python_types.ensure_bool(
        param=email_on_start,
        param_name="email_on_start",
    )
    validate_python_types.ensure_bool(
        param=email_on_finish,
        param_name="email_on_finish",
    )
    validate_python_types.ensure_bool(
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


def _build_slurm_script(
    *,
    directives: list[str] | None,
    tag_name: str,
    partition_name: str | None,
    num_cpus: int | None,
    memory_gb: int | None,
    wall_time_hours: int | None,
    email_address: str | None,
    email_events_string: str,
    prep_command: str | None,
    main_command: str,
    post_command: str | None,
    always_run_post: bool,
) -> list[str]:
    file_lines: list[str] = []
    ## --- slurm header
    if directives is None:
        assert partition_name is not None
        assert num_cpus is not None
        assert wall_time_hours is not None
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={tag_name}",
            f"#SBATCH --partition={partition_name}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={num_cpus}",
            f"#SBATCH --time={wall_time_hours:02}:00:00",
            "#SBATCH --output=%x_%j.out",
            "#SBATCH --error=%x_%j.err",
        ]
        if memory_gb is not None:
            directives.insert(5, f"#SBATCH --mem={memory_gb}G")
    file_lines += directives
    if email_address is not None:
        file_lines += [
            f"#SBATCH --mail-user={email_address}",
            f"#SBATCH --mail-type={email_events_string}",
        ]
    ## --- shell setup
    file_lines += [
        "",
        "set -euo pipefail",
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


def create_slurm_job_script(
    *,
    directory: str | Path,
    file_name: str,
    directives: list[str] | None = None,
    main_command: str,
    tag_name: str,
    partition_name: str | None = None,
    num_cpus: int | None = None,
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
    Create a SLURM job script with caller-provided directives or generic resources.

    Parameters
    ---
    - `directory`:
        Directory where the job script file will be written.

    - `file_name`:
        Filename for the job script; must end with `.sh`.

    - `directives`:
        Ordered SLURM header lines to write at the top of the script. This should
        include the shebang and any `#SBATCH ...` directives required by the
        target system. When omitted, a generic SLURM header is built from
        `partition_name`, `num_cpus`, `memory_gb`, and `wall_time_hours`.

    - `main_command`:
        Primary workload command. Its exit code is captured and used as the
        script's final exit code.

    - `tag_name`:
        SLURM job name used by the generic header builder and as the prefix for log files.

    - `partition_name`, `num_cpus`, `memory_gb`, `wall_time_hours`:
        Generic SLURM resource settings used when `directives` are not supplied.
    """
    _ensure_inputs(
        directory=directory,
        file_name=file_name,
        directives=directives,
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
    email_events: list[str] = ["FAIL"]
    if email_on_start:
        email_events.append("BEGIN")
    if email_on_finish:
        email_events.append("END")
    email_events_string = ",".join(email_events)
    file_path = Path(directory) / file_name
    file_path = _ensure_path_is_valid(file_path=file_path)
    file_lines = _build_slurm_script(
        directives=directives,
        tag_name=tag_name,
        partition_name=partition_name,
        num_cpus=num_cpus,
        memory_gb=memory_gb,
        wall_time_hours=wall_time_hours,
        email_address=email_address,
        email_events_string=email_events_string,
        prep_command=prep_command,
        main_command=main_command,
        post_command=post_command,
        always_run_post=always_run_post,
    )
    file_path.write_text("\n".join(file_lines) + "\n")
    if verbose:
        manage_log.log_action(
            title="Create SLURM job",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="Wrote job script.",
            notes={
                "file": str(file_path),
                "tag_name": tag_name,
            },
        )
    return file_path


## } MODULE
