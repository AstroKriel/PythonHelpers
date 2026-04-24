## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import shlex
import subprocess

from dataclasses import dataclass
from pathlib import Path

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class CommandOutcome:
    """
    Result of a completed shell command execution.

    Fields
    ---
    - `command`:
        The command string that was executed.

    - `working_directory`:
        Working directory used for the command; `None` if not specified.

    - `exit_code`:
        Process exit code; 0 indicates success.

    - `stdout`:
        Captured standard output, or None if output was streamed to console.

    - `stderr`:
        Captured standard error, or None if output was streamed to console.
    """

    command: str
    working_directory: Path | None
    exit_code: int
    stdout: str | None
    stderr: str | None

    @property
    def succeeded(
        self,
    ) -> bool:
        return self.exit_code == 0


##
## === FUNCTIONS
##


def execute_shell_command(
    command: str,
    *,
    working_directory: str | Path | None = None,
    timeout_seconds: float = 60,
    use_shell: bool = False,
    capture_output: bool = False,
    raise_on_error: bool = True,
) -> CommandOutcome:
    """
    Run a shell command and return its outcome.

    Parameters
    ---
    - `working_directory`:
        Directory to run the command in; `None` uses the caller's working directory.

    - `timeout_seconds`:
        Seconds before the command is killed with `TimeoutExpired`.

    - `use_shell`:
        Pass the command to the system shell instead of splitting it with `shlex`.
        Only safe with fully trusted, hardcoded strings -- never use with user input.

    - `capture_output`:
        `True` stores stdout/stderr in the outcome; `False` streams them to the console.

    - `raise_on_error`:
        Raise `RuntimeError` on non-zero exit code when `True`.
    """
    resolved_directory = Path(working_directory) if working_directory is not None else None
    try:
        completed = subprocess.run(
            command if use_shell else shlex.split(command),
            cwd=resolved_directory,
            timeout=timeout_seconds,
            capture_output=capture_output,
            shell=use_shell,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"command not found: {command}.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"command timed out after {timeout_seconds}s: {command}.") from error
    command_outcome = CommandOutcome(
        command=command,
        working_directory=resolved_directory,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if raise_on_error and not command_outcome.succeeded:
        message = f"Command failed with exit code {command_outcome.exit_code}: {command_outcome.command}"
        if command_outcome.stdout:
            message += f"\nstdout:\n{command_outcome.stdout.strip()}"
        if command_outcome.stderr:
            message += f"\nstderr:\n{command_outcome.stderr.strip()}"
        raise RuntimeError(message)
    return command_outcome


## } MODULE
