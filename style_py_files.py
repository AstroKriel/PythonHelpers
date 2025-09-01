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

##
## === GLOBAL PARAMS ===
##

PROJECT_DIR = Path(__file__).resolve().parent
STYLE_FILE = PROJECT_DIR / ".style.yapf"
DEFAULT_FOLDERS = ("src", "playground", "tests", "scripts")
PY_DEPENDENCIES = ["add-trailing-comma", "yapf"]

##
## === HELPER FUNCTIONS ===
##


def ensure_running_from_project_root() -> None:
    # verifies the script is PLACED in the project root (does not force running from it)
    if not (PROJECT_DIR / "pyproject.toml").exists() and not (PROJECT_DIR / ".git").exists():
        sys.exit(
            "ERROR: this script must be placed in the project root (where `pyproject.toml` or `.git` exists)"
        )


def ensure_uv_available() -> None:
    if shutil.which("uv") is None:
        sys.exit("ERROR: `uv` not found in PATH. install uv first.")


def ensure_tools_installed() -> None:
    py_dependencies = " ".join(PY_DEPENDENCIES)
    shell_manager.execute_shell_command(
        f"uv add {py_dependencies}",
        working_directory=PROJECT_DIR,
        timeout_seconds=120,
    )


def collect_py_files(directories: Iterable[Path]) -> list[Path]:
    file_paths: list[Path] = []
    for path in directories:
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".py":
            file_paths.append(path)
            continue
        for directory, _dirnames, file_names in os.walk(path):
            for file_name in file_names:
                if file_name.endswith(".py"):
                    file_paths.append(Path(directory) / file_name)
    return file_paths


def apply_trailing_commas(file_paths: list[Path]) -> None:
    if not file_paths:
        return
    print(f"adding trailing commas where safe ({len(file_paths)} files)...")
    # run per-file to avoid very long command lines
    for file_path in file_paths:
        shell_manager.execute_shell_command(
            f'uv run add-trailing-comma "{file_path}"',
            working_directory=PROJECT_DIR,
            timeout_seconds=120,
        )


def apply_yapf_style(targets: list[Path]) -> None:
    if not targets:
        return
    if not STYLE_FILE.exists():
        sys.exit(f"ERROR: `{STYLE_FILE}` was not found. add a `.style.yapf` in the project root.")
    print(f"running yapf (style={STYLE_FILE}) on {len(targets)} target(s)...")
    quoted_targets = " ".join(f'"{t}"' for t in targets)
    shell_manager.execute_shell_command(
        f'uv run yapf -ir --verbose --style "{STYLE_FILE}" {quoted_targets}',
        working_directory=PROJECT_DIR,
        timeout_seconds=300,
    )


def format_project(targets: list[str] | None = None) -> int:
    ensure_running_from_project_root()
    ensure_uv_available()
    ensure_tools_installed()

    raw_targets = targets or list(DEFAULT_FOLDERS)
    resolved_targets = [Path(target).resolve() for target in raw_targets]
    file_paths = collect_py_files(resolved_targets)

    if not file_paths:
        print("no python files were found under:", ", ".join(map(str, resolved_targets)))
        return 0

    apply_trailing_commas(file_paths)
    apply_yapf_style(resolved_targets)
    print("finished.")
    return 0


##
## === MAIN PROGRAM ===
##


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="apply trailing-commas rule and yapf styling to selected targets."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "folders or files to format (default: src, playground, tests, scripts). "
            "paths are interpreted relative to your current working directory; "
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
