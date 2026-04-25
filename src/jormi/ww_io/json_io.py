## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import copy
import json

from pathlib import Path
from typing import Any

## third-party
import numpy

## local
from jormi.ww_dicts import merge_dicts
from jormi.ww_fns import fn_decorators
from jormi.ww_io import manage_log
from jormi.ww_validation import validate_types

##
## === FUNCTIONS
##


def _ensure_path_is_valid(
    file_path: str | Path,
) -> Path:
    """Ensure `file_path` is a valid .json path and return it as an absolute Path."""
    validate_types.ensure_not_none(
        param=file_path,
        param_name="file_path",
    )
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".json":
        raise ValueError(f"file must end with a `.json` extension: {file_path}.")
    return file_path


def read_json_file_into_dict(
    file_path: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    validate_types.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No json-file found: {file_path}")
    if verbose:
        manage_log.log_task(f"Reading json-file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file_pointer:
        data = json.load(file_pointer)
    validate_types.ensure_dict(
        param=data,
        param_name="JSON root object",
        allow_none=False,
    )
    return copy.deepcopy(data)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, arrays, and WarnIfUnused wrappers."""

    def default(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        param: Any,
    ) -> Any:
        if isinstance(param, numpy.integer):
            return int(param)
        if isinstance(param, numpy.floating):
            return float(param)
        if isinstance(param, numpy.bool_):
            return bool(param)
        if isinstance(param, numpy.ndarray):
            return param.tolist()
        if isinstance(param, fn_decorators.WarnIfUnused):
            return param.unwrap()
        return super().default(param)


def save_dict_to_json_file(
    file_path: str | Path,
    input_dict: dict[str, Any],
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    validate_types.ensure_bool(
        param=overwrite,
        param_name="overwrite",
        allow_none=False,
    )
    validate_types.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    validate_types.ensure_dict(
        param=input_dict,
        param_name="input_dict",
        allow_none=False,
    )
    file_exists = file_path.is_file()
    if file_exists and not overwrite:
        _add_dict_to_json_file(
            file_path=file_path,
            input_dict=input_dict,
        )
        if verbose:
            manage_log.log_action(
                title="Save JSON file",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message="Updated json-file (merged dictionaries).",
                notes={
                    "file": str(file_path),
                    "mode": "merge",
                },
            )
    else:
        _dump_dict_to_json(
            file_path=file_path,
            input_dict=input_dict,
        )
        if verbose:
            mode = "overwrite" if file_exists and overwrite else "create"
            message = ("Overwrote existing json-file." if mode == "overwrite" else "Saved new json-file.")
            manage_log.log_action(
                title="Save JSON file",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message=message,
                notes={
                    "file": str(file_path),
                    "mode": mode,
                },
            )


def _dump_dict_to_json(
    file_path: str | Path,
    input_dict: dict[str, Any],
) -> None:
    file_path = _ensure_path_is_valid(file_path)
    with open(file_path, "w", encoding="utf-8") as file_pointer:
        json.dump(
            obj=input_dict,
            fp=file_pointer,
            cls=NumpyEncoder,
            sort_keys=True,
            indent=2,
        )


def _add_dict_to_json_file(
    file_path: str | Path,
    input_dict: dict[str, Any],
) -> None:
    """Merge `input_dict` into an existing JSON file."""
    old_dict = read_json_file_into_dict(
        file_path=file_path,
        verbose=False,
    )
    merged_dict = merge_dicts(
        dict_a=old_dict,
        dict_b=input_dict,
    )
    _dump_dict_to_json(
        file_path=file_path,
        input_dict=merged_dict,
    )


## } MODULE
