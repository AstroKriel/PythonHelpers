## { MODULE

##
## === DEPENDENCIES
##

import json
import copy
import numpy

from pathlib import Path

from jormi.ww_io import io_manager, log_manager
from jormi.ww_types import type_manager
from jormi.utils import dict_utils

##
## === FUNCTIONS
##


def _ensure_path_is_valid(
    file_path: str | Path,
) -> Path:
    """Ensure `file_path` is a valid .json path and return it as an absolute Path."""
    type_manager.ensure_not_none(
        param=file_path,
        param_name="file_path",
    )
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".json":
        raise ValueError(f"File should end with a .json extension: {file_path}")
    return file_path


def read_json_file_into_dict(
    file_path: str | Path,
    *,
    verbose: bool = True,
):
    type_manager.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    if not io_manager.does_file_exist(file_path=file_path):
        raise FileNotFoundError(f"No json-file found: {file_path}")
    if verbose:
        log_manager.log_task(f"Reading json-file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file_pointer:
        data = json.load(file_pointer)
    type_manager.ensure_dict(
        param=data,
        param_name="JSON root object",
        allow_none=False,
    )
    return copy.deepcopy(data)


class NumpyEncoder(json.JSONEncoder):

    def default(  # type: ignore
        self,
        param,
    ):
        if isinstance(param, numpy.integer):
            return int(param)
        elif isinstance(param, numpy.floating):
            return float(param)
        elif isinstance(param, numpy.bool_):
            return bool(param)
        elif isinstance(param, numpy.ndarray):
            return param.tolist()
        return super().default(param)


def save_dict_to_json_file(
    file_path: str | Path,
    input_dict: dict,
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    type_manager.ensure_bool(
        param=overwrite,
        param_name="overwrite",
        allow_none=False,
    )
    type_manager.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    type_manager.ensure_dict(
        param=input_dict,
        param_name="input_dict",
        allow_none=False,
    )
    file_exists = io_manager.does_file_exist(file_path=file_path)
    if file_exists and not overwrite:
        _add_dict_to_json_file(
            file_path=file_path,
            input_dict=input_dict,
        )
        if verbose:
            log_manager.log_action(
                title="Save JSON file",
                outcome=log_manager.ActionOutcome.SUCCESS,
                message="Updated json-file (merged dictionaries).",
                notes={
                    "file": str(file_path),
                    "mode": "merge",
                },
            )
    else:
        _create_json_file_from_dict(
            file_path=file_path,
            input_dict=input_dict,
        )
        if verbose:
            mode = "overwrite" if file_exists and overwrite else "create"
            message = ("Overwrote existing json-file." if mode == "overwrite" else "Saved new json-file.")
            log_manager.log_action(
                title="Save JSON file",
                outcome=log_manager.ActionOutcome.SUCCESS,
                message=message,
                notes={
                    "file": str(file_path),
                    "mode": mode,
                },
            )


def _dump_dict_to_json(
    file_path: str | Path,
    input_dict: dict,
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


def _create_json_file_from_dict(
    file_path: str | Path,
    input_dict: dict,
) -> None:
    """Create (or overwrite) a JSON file from `input_dict`."""
    _dump_dict_to_json(
        file_path=file_path,
        input_dict=input_dict,
    )


def _add_dict_to_json_file(
    file_path: str | Path,
    input_dict: dict,
) -> None:
    """Merge `input_dict` into an existing JSON file."""
    old_dict = read_json_file_into_dict(
        file_path=file_path,
        verbose=False,
    )
    merged_dict = dict_utils.merge_dicts(
        dict_a=old_dict,
        dict_b=input_dict,
    )
    _dump_dict_to_json(
        file_path=file_path,
        input_dict=merged_dict,
    )


## } MODULE
