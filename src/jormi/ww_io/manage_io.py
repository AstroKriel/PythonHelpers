## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import shutil
from pathlib import Path

## local
from jormi.ww_io import manage_log

##
## === FUNCTIONS
##


def create_directory(
    directory: str | Path,
    *,
    verbose: bool = True,
) -> None:
    directory = Path(directory).resolve(strict=False)
    if not directory.is_dir():
        directory.mkdir(parents=True)
        if verbose:
            manage_log.log_action(
                title="Initialise directory",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message="Created directory.",
                notes={"directory": str(directory)},
            )
    elif verbose:
        manage_log.log_action(
            title="Initialise directory",
            outcome=manage_log.ActionOutcome.SKIPPED,
            message="Directory already exists.",
            notes={"directory": str(directory)},
        )


def _resolve_file_paths(
    *,
    directory_from: str | Path,
    directory_to: str | Path,
    file_name: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    directory_from = Path(directory_from)
    if not directory_from.is_dir():
        raise NotADirectoryError(f"directory does not exist: {directory_from}.")
    file_path_from = directory_from / file_name
    if not file_path_from.is_file():
        raise FileNotFoundError(f"file does not exist: {file_path_from}.")
    directory_to = Path(directory_to)
    file_path_to = directory_to / file_name
    if not overwrite and file_path_to.is_file():
        raise FileExistsError(f"file already exists: {file_path_to}.")
    return file_path_from, file_path_to


def _log_file_action(
    *,
    action: str,
    file_name: str,
    directory_from: str | Path,
    directory_to: str | Path | None = None,
    is_dry_run: bool,
) -> None:
    notes: dict[str, str] = {
        "file": file_name,
        "source": str(directory_from),
    }
    if directory_to is not None:
        notes["target"] = str(directory_to)
    manage_log.log_action(
        title=action,
        outcome=(
            manage_log.ActionOutcome.SKIPPED
            if is_dry_run
            else manage_log.ActionOutcome.SUCCESS
        ),
        message="",
        notes=notes,
    )


def copy_file(
    directory_from: str | Path,
    directory_to: str | Path,
    file_name: str,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    file_path_from, file_path_to = _resolve_file_paths(
        directory_from=directory_from,
        directory_to=directory_to,
        file_name=file_name,
        overwrite=overwrite,
    )
    if dry_run:
        if not Path(directory_to).is_dir():
            manage_log.log_action(
                title="Create directory",
                outcome=manage_log.ActionOutcome.SKIPPED,
                message="Dry-run: would create directory.",
                notes={"directory": str(directory_to)},
            )
    else:
        create_directory(directory=directory_to, verbose=False)
        shutil.copy2(src=file_path_from, dst=file_path_to)
    if verbose or dry_run:
        _log_file_action(
            action="Copy file",
            file_name=file_name,
            directory_from=directory_from,
            directory_to=directory_to,
            is_dry_run=dry_run,
        )


def move_file(
    directory_from: str | Path,
    directory_to: str | Path,
    file_name: str,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    file_path_from, file_path_to = _resolve_file_paths(
        directory_from=directory_from,
        directory_to=directory_to,
        file_name=file_name,
        overwrite=overwrite,
    )
    if dry_run:
        if not Path(directory_to).is_dir():
            manage_log.log_action(
                title="Create directory",
                outcome=manage_log.ActionOutcome.SKIPPED,
                message="Dry-run: would create directory.",
                notes={"directory": str(directory_to)},
            )
    else:
        create_directory(directory=directory_to, verbose=False)
        shutil.move(src=file_path_from, dst=file_path_to)
    if verbose or dry_run:
        _log_file_action(
            action="Move file",
            file_name=file_name,
            directory_from=directory_from,
            directory_to=directory_to,
            is_dry_run=dry_run,
        )


def delete_file(
    directory: str | Path,
    file_name: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"directory does not exist: {directory}.")
    file_path = directory / file_name
    if not dry_run:
        file_path.unlink()
    if verbose or dry_run:
        _log_file_action(
            action="Delete file",
            file_name=file_name,
            directory_from=directory,
            is_dry_run=dry_run,
        )


def filter_directory(
    directory: str | Path,
    *,
    req_include_words: str | list[str] | None = None,
    req_exclude_words: str | list[str] | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    delimiter: str = "_",
    num_parts: int | None = None,
    include_files: bool = True,
    include_folders: bool = True,
) -> list[Path]:
    """
    Filter items in a directory by name criteria, returning a sorted list of matching paths.

    Parameters
    ---
    - `req_include_words`:
        Words that must all appear in the item name; `None` means no requirement.

    - `req_exclude_words`:
        Words that must not appear in the item name; `None` means no exclusion.

    - `prefix`:
        Required name prefix; `None` to skip.

    - `suffix`:
        Required name suffix; `None` to skip.

    - `delimiter`:
        Character used to split the name into parts for `num_parts` checks.

    - `num_parts`:
        Required number of name parts after splitting by `delimiter`; `None` to skip.

    - `include_files`:
        Whether to include files in results.

    - `include_folders`:
        Whether to include directories in results. At least one of `include_files` or
        `include_folders` must be `True`.
    """

    def _to_word_list(
        value: str | list[str] | None,
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value  # pyright: ignore[reportReturnType]

    if not (include_files or include_folders):
        raise ValueError(
            "At least one of `include_files` or `include_folders` must be provided.",
        )
    directory = Path(directory).absolute()
    if not directory.is_dir():
        raise NotADirectoryError(f"directory does not exist: {directory}.")
    include_list = _to_word_list(req_include_words)
    exclude_list = _to_word_list(req_exclude_words)
    matched: list[Path] = []
    for item in directory.iterdir():
        if item.is_file() and not include_files:
            continue
        if item.is_dir() and not include_folders:
            continue
        name = item.name
        if include_list and not all(word in name for word in include_list):
            continue
        if exclude_list and any(word in name for word in exclude_list):
            continue
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        if num_parts is not None and len(name.split(delimiter)) != num_parts:
            continue
        matched.append(item)
    return sorted(matched)


## } MODULE
