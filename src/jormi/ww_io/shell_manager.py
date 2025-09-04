## { MODULE

##
## === DEPENDENCIES ===
##

import shlex
import subprocess
from pathlib import Path
from dataclasses import dataclass

##
## === RESULT TYPE ===
##

@dataclass(frozen=True)
class ShellCommandResult:
    display_command: str
    working_directory: str | None
    exit_code: int
    stdout: str | None
    stderr: str | None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


##
## === FUNCTION ===
##

def execute_shell_command(
    command: str,
    *,
    working_directory: str | Path | None = None,
    timeout_seconds: float = 60,
    use_shell: bool = False,
    capture_output: bool = False,
    raise_on_error: bool = True,
) -> ShellCommandResult:
    """
    Run a `command` and either stream it to the console or capture the output.

    capture_output:
      - True: return stdout/stderr in the result (good for parsing or summarizing)
      - False: stream output live to console (good for long-running, human-facing ops)
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
    result = ShellCommandResult(
        display_command=command,
        working_directory=working_directory,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if raise_on_error and not result.succeeded:
        message = f"Command failed with exit code {result.exit_code}: {result.display_command}"
        if result.stdout:
            message += f"\nstdout:\n{result.stdout.strip()}"
        if result.stderr:
            message += f"\nstderr:\n{result.stderr.strip()}"
        raise RuntimeError(message)
    return result


## } MODULE
