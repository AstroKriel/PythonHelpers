## { SCRIPT

##
## === DEPENDENCIES ===
##

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Iterable
from jormi.ww_io import shell_manager
from jormi.utils import logging_utils

##
## === GLOBAL PARAMS ===
##

PROJECT_DIR = Path(__file__).resolve().parent
STYLE_FILE_NAME = ".style.yapf"
STYLE_FILE_PATH = PROJECT_DIR / STYLE_FILE_NAME
PY_DEPENDENCIES = ("add-trailing-comma", "yapf")
EXTENSIONS = {".py"}
FILES_TO_IGNORE = {}
DIRS_TO_IGNORE = (
    "__pycache__",
    ".venv",
    ".git",
    "build",
    "dist",
    ".eggs",
)

##
## === LOGGING HELPERS ===
##


def log_info(text: str) -> None:
    logging_utils.render_line(
        logging_utils.Message(text, message_type=logging_utils.MessageType.GENERAL),
        show_time=True,
    )


def log_action(text: str, *, outcome: logging_utils.ActionOutcome) -> None:
    logging_utils.render_line(
        logging_utils.Message(
            text,
            message_type=logging_utils.MessageType.ACTION,
            action_outcome=outcome,
        ),
        show_time=True,
    )


##
## === HELPER FUNCTIONS ===
##


def ensure_script_is_in_project_root() -> None:
    # Verifies the script is PLACED in the project root (does not force running from it)
    if not (PROJECT_DIR / "pyproject.toml").exists() and not (PROJECT_DIR / ".git").exists():
        log_action(
            f"This script is not placed in the project root (expected `pyproject.toml` or `.git` next to `{PROJECT_DIR}`)",
            outcome=logging_utils.ActionOutcome.FAILURE,
        )
        sys.exit(1)


def ensure_uv_available() -> None:
    if shutil.which("uv") is None:
        log_action(
            "`uv` not found in PATH. Install uv first.",
            outcome=logging_utils.ActionOutcome.FAILURE,
        )
        sys.exit(1)
    log_action("Found `uv`", outcome=logging_utils.ActionOutcome.SUCCESS)


def ensure_tools_installed() -> None:
    py_dependencies = " ".join(PY_DEPENDENCIES)
    log_info(f"Ensuring tools are available: {py_dependencies}")
    shell_manager.execute_shell_command(
        f"uv add {py_dependencies}",
        working_directory=PROJECT_DIR,
        timeout_seconds=120,
    )
    log_action(
        "Tool installation / up-to-date check completed",
        outcome=logging_utils.ActionOutcome.SUCCESS,
    )


def _should_ignore_dirname(dir_name: str) -> bool:
    return dir_name in DIRS_TO_IGNORE


def _should_ignore_file(path: Path) -> bool:
    if path.name in FILES_TO_IGNORE:
        return True
    if path.suffix not in EXTENSIONS:
        return True
    if any(path_part in DIRS_TO_IGNORE for path_part in path.parts):
        return True
    return False


def collect_py_files(targets: Iterable[Path]) -> list[Path]:
    file_paths: list[Path] = []
    for path in targets:
        if not path.exists():
            continue
        if path.is_file():
            if not _should_ignore_file(path):
                file_paths.append(path)
            continue
        for dir_path, dir_names, file_names in os.walk(path, topdown=True):
            ## prune directories in-place so os.walk does not descend into them
            dir_names[:] = [
                dir_name for dir_name in dir_names if not _should_ignore_dirname(dir_name)
            ]
            for filename in file_names:
                full_path = Path(dir_path) / filename
                if _should_ignore_file(full_path):
                    continue
                file_paths.append(full_path)
    file_paths.sort()
    return file_paths


def apply_trailing_commas(file_paths: list[Path]) -> None:
    if not file_paths:
        log_info("No Python files to update for trailing commas")
        return
    log_info(f"Adding trailing commas where safe ({len(file_paths)} files)")
    for file_path in file_paths:
        shell_manager.execute_shell_command(
            f'uv run add-trailing-comma --exit-zero-even-if-changed "{file_path}"',
            working_directory=PROJECT_DIR,
            timeout_seconds=120,
            show_output=True,
        )
    log_action("Trailing-commas pass completed", outcome=logging_utils.ActionOutcome.SUCCESS)


def apply_yapf_style(file_paths: list[Path]) -> None:
    if not file_paths:
        log_info("No files for YAPF")
        return
    if not STYLE_FILE_PATH.exists():
        log_action(
            f"`{STYLE_FILE_NAME}` was not found in the project root: {PROJECT_DIR}",
            outcome=logging_utils.ActionOutcome.FAILURE,
        )
        sys.exit(1)
    log_info(f"Running YAPF-styling on {len(file_paths)} file(s)")
    for file_path in file_paths:
        shell_manager.execute_shell_command(
            f'uv run yapf -i --verbose --style "{STYLE_FILE_PATH}" "{file_path}"',
            working_directory=PROJECT_DIR,
            timeout_seconds=300,
            show_output=True,
        )
    log_action("YAPF formatting completed", outcome=logging_utils.ActionOutcome.SUCCESS)


##
## === MAIN ROUTINE ===
##


def format_project(targets: list[str] | None = None) -> int:
    log_info("Formatting project")
    ensure_script_is_in_project_root()
    ensure_uv_available()
    ensure_tools_installed()
    if not targets:
        ## work from the project root
        resolved_targets = [PROJECT_DIR]
    else:
        ## work relative to the cwd
        resolved_targets = [Path(target).resolve() for target in targets]
    file_paths = collect_py_files(resolved_targets)
    log_info(f"Discovered {len(file_paths)} Python files across {len(resolved_targets)} target(s)")
    if not file_paths:
        log_info("No Python files were found under: " + ", ".join(map(str, resolved_targets)))
        log_action("Nothing to do", outcome=logging_utils.ActionOutcome.SKIPPED)
        return 0
    apply_trailing_commas(file_paths)
    apply_yapf_style(file_paths)
    log_action("Formatting finished", outcome=logging_utils.ActionOutcome.SUCCESS)
    return 0


##
## === MAIN PROGRAM ===
##


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Apply trailing-commas rule and YAPF styling to selected targets.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "Folders or files to format. "
            "If none are provided, the whole project (PROJECT_DIR) is scanned and formatted. "
            "If provided, relevant targets are resolved relative to your current working directory; "
            "absolute paths are also accepted."
        ),
    )
    args = parser.parse_args(argv)
    return format_project(args.targets)


##
## === ENTRY POINT ===
##

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

## } SCRIPT
