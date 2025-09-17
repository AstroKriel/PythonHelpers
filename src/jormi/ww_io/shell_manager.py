## { MODULE

##
## === DEPENDENCIES
##

import shlex
import subprocess
from pathlib import Path
from dataclasses import dataclass

##
## === RESULT TYPE
##


@dataclass(frozen=True)
class CommandOutcome:
    command: str
    working_directory: str | None
    exit_code: int
    stdout: str | None
    stderr: str | None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


##
## === FUNCTION
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
    Run a `command` and either stream it to the console or capture the output.

    `capture_output`:
      - True: store stdout/stderr in the outcome
      - False: stream the output to the console buffer
    """
    if isinstance(working_directory, Path):
        working_directory = str(working_directory)
    try:
        completed = subprocess.run(
            command if use_shell else shlex.split(command),
            cwd=working_directory,
            timeout=timeout_seconds,
            capture_output=capture_output,
            shell=use_shell,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Command not found: {command}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command timed out after {timeout_seconds}s: {command}") from error
    command_outcome = CommandOutcome(
        command=command,
        working_directory=working_directory,
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
