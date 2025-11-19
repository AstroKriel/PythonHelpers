## { MODULE

##
## === DEPENDENCIES
##

import numpy
import shutil
import inspect

from pathlib import Path

from jormi.utils import list_utils
from jormi.ww_io import log_manager

##
## === FUNCTIONS
##


def get_caller_directory() -> Path:
    """Get the directory of the script that invoked this function."""
    caller_frame = inspect.stack()[1]
    caller_file_path = caller_frame.filename
    return Path(caller_file_path).resolve().parent


def combine_file_path_parts(
    file_path_parts: list[str | Path],
) -> Path:
    return Path(
        *list_utils.flatten_list(
            list_utils.filter_out_nones(file_path_parts),
        ),
    ).absolute()


def resolve_file_path(
    file_path: str | Path | None = None,
    directory: str | Path | None = None,
    file_name: str | None = None,
) -> Path:
    if file_path is None:
        missing_params = []
        if directory is None:
            missing_params.append("directory")
        if file_name is None:
            missing_params.append("file_name")
        if missing_params:
            missing_params_string = list_utils.as_string(missing_params)
            raise ValueError(
                "You have not provided enough information about the file and where it is."
                f" You are missing: {missing_params_string}."
                " Alternatively, provide `file_path` directly.",
            )
        file_path = combine_file_path_parts(
            list_utils.filter_out_nones([directory, file_name]),
        )
    else:
        file_path = Path(file_path).absolute()
    return file_path


def does_directory_exist(
    directory: str | Path,
    raise_error: bool = False,
) -> bool:
    directory = Path(directory).absolute()
    result = directory.is_dir()
    if not result and raise_error:
        raise NotADirectoryError(f"Directory does not exist: {directory}")
    return result


def init_directory(
    directory: str | Path,
    verbose: bool = True,
) -> None:
    directory = Path(directory).resolve(strict=False)
    if not does_directory_exist(directory):
        directory.mkdir(parents=True)
        if verbose:
            log_manager.log_action(
                title="Initialise directory",
                outcome=log_manager.ActionOutcome.SUCCESS,
                message="Created directory.",
                notes={"directory": str(directory)},
            )
    elif verbose:
        log_manager.log_action(
            title="Initialise directory",
            outcome=log_manager.ActionOutcome.SKIPPED,
            message="Directory already exists.",
            notes={"directory": str(directory)},
        )


def does_file_exist(
    file_path: str | Path | None = None,
    directory: str | Path | None = None,
    file_name: str | None = None,
    raise_error: bool = False,
) -> bool:
    file_path = resolve_file_path(
        file_path=file_path,
        directory=directory,
        file_name=file_name,
    )
    file_path_exists = file_path.is_file()
    if not file_path_exists and raise_error:
        raise FileNotFoundError(f"File does not exist: {file_path}")
    return file_path_exists


def _resolve_and_validate_file_operation(
    directory_from: str | Path,
    directory_to: str | Path,
    file_name: str,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    does_directory_exist(
        directory=directory_from,
        raise_error=True,
    )
    file_path_from = combine_file_path_parts([directory_from, file_name])
    does_file_exist(
        file_path=file_path_from,
        raise_error=True,
    )
    if not does_directory_exist(directory=directory_to):
        if not dry_run:
            init_directory(
                directory=directory_to,
                verbose=False,
            )
        else:
            log_manager.log_action(
                title="Create directory",
                outcome=log_manager.ActionOutcome.SKIPPED,
                message="Dry-run: Would create directory.",
                notes={"directory": str(directory_to)},
            )
    file_path_to = combine_file_path_parts([directory_to, file_name])
    file_exists = does_file_exist(
        file_path=file_path_to,
        raise_error=False,
    )
    if not overwrite and file_exists:
        raise FileExistsError(f"File already exists: {file_path_to}")
    return file_path_from, file_path_to


def _log_file_action(
    action: str,
    file_name: str,
    directory_from: str | Path,
    directory_to: str | Path | None = None,
    *,
    is_dry_run: bool,
) -> None:
    """Log a file operation using the log_manager API."""
    notes: dict[str, str] = {
        "file": file_name,
        "source": str(directory_from),
    }
    if directory_to is not None:
        notes["target"] = str(directory_to)
    log_manager.log_action(
        title=action,
        outcome=(log_manager.ActionOutcome.SKIPPED if is_dry_run else log_manager.ActionOutcome.SUCCESS),
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
    file_path_from, file_path_to = _resolve_and_validate_file_operation(
        directory_from=directory_from,
        directory_to=directory_to,
        file_name=file_name,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if not dry_run:
        shutil.copy(
            src=file_path_from,
            dst=file_path_to,
        )
        shutil.copymode(
            src=file_path_from,
            dst=file_path_to,
        )
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
    file_path_from, file_path_to = _resolve_and_validate_file_operation(
        directory_from=directory_from,
        directory_to=directory_to,
        file_name=file_name,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    if not dry_run:
        shutil.move(
            src=file_path_from,
            dst=file_path_to,
        )
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
    does_directory_exist(
        directory=directory,
        raise_error=True,
    )
    file_path = combine_file_path_parts([directory, file_name])
    if not dry_run:
        file_path.unlink()
    if verbose or dry_run:
        _log_file_action(
            action="Delete file",
            file_name=file_name,
            directory_from=directory,
            is_dry_run=dry_run,
        )


class ItemFilter:

    def __init__(
        self,
        *,
        req_include_words: str | list[str] | None = None,
        req_exclude_words: str | list[str] | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        delimiter: str = "_",
        num_parts: int | None = None,
        index_of_value: int | None = None,
        min_value: int | float = 0,
        max_value: int | float = numpy.inf,
        include_files: bool = True,
        include_folders: bool = True,
    ):
        self.req_include_words = self._to_list(req_include_words)
        self.req_exclude_words = self._to_list(req_exclude_words)
        self.prefix = prefix
        self.suffix = suffix
        self.delimiter = delimiter
        self.num_parts = num_parts
        self.index_of_value = index_of_value
        self.min_value = min_value
        self.max_value = max_value
        self.include_files = include_files
        self.include_folders = include_folders
        self._validate_inputs()

    def _to_list(
        self,
        value,
    ):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        raise ValueError("Expected a string or list of strings.")

    def _validate_inputs(
        self,
    ):
        if not (self.include_files or self.include_folders):
            raise ValueError(
                "At least one of `include_files` or `include_folders` must be enabled.",
            )
        if not isinstance(self.min_value, (int, float)) or not isinstance(
                self.max_value,
            (int, float),
        ):
            raise TypeError("`min_value` and `max_value` must be numbers.")
        if self.min_value > self.max_value:
            raise ValueError("`min_value` cannot be greater than `max_value`.")

    def _meets_criteria(
        self,
        item_path: Path,
    ) -> bool:
        if item_path.is_file() and not self.include_files:
            return False
        if item_path.is_dir() and not self.include_folders:
            return False
        item_name = item_path.name
        all_include_words_present = all(
            req_include_word in item_name for req_include_word in self.req_include_words
        )
        any_exclude_words_present = any(
            req_exclude_word in item_name for req_exclude_word in self.req_exclude_words
        )
        if self.req_include_words and not all_include_words_present:
            return False
        if self.req_exclude_words and any_exclude_words_present:
            return False
        if self.prefix and not item_name.startswith(self.prefix):
            return False
        if self.suffix and not item_name.endswith(self.suffix):
            return False
        name_parts = item_name.split(self.delimiter)
        if (self.num_parts is not None) and (len(name_parts) != self.num_parts):
            return False
        if self.index_of_value is not None:
            if len(name_parts) < abs(self.index_of_value):
                return False
            try:
                value = int(name_parts[self.index_of_value])
            except ValueError:
                return False
            if (value < self.min_value) or (self.max_value < value):
                return False
        return True

    def filter(
        self,
        directory: str | Path,
    ) -> list[Path]:
        """Filter item names in the given directory based on current criteria."""
        directory = Path(directory).absolute()
        does_directory_exist(directory, raise_error=True)
        return sorted(item for item in directory.iterdir() if self._meets_criteria(item))


## } MODULE
