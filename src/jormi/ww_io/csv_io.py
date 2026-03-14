## { MODULE

##
## === DEPENDENCIES
##

import csv

from pathlib import Path

from jormi.ww_io import io_manager, log_manager
from jormi.ww_types import type_checks

##
## === FUNCTIONS
##


def _ensure_path_is_valid(
    file_path: str | Path,
) -> Path:
    """Ensure `file_path` is a valid .csv path and return it as an absolute Path."""
    type_checks.ensure_not_none(
        param=file_path,
        param_name="file_path",
    )
    file_path = Path(file_path).absolute()
    if file_path.suffix != ".csv":
        raise ValueError(f"File should end with a .csv extension: {file_path}")
    return file_path


def _validate_input_dict(
    input_dict: dict,
) -> None:
    """Validate that `input_dict` is a dict with string keys."""
    type_checks.ensure_dict(
        param=input_dict,
        param_name="input_dict",
        allow_none=False,
    )
    if not input_dict:
        return
    keys_as_list = list(input_dict.keys())
    type_checks.ensure_list_of_strings(
        param=keys_as_list,
        param_name="input_dict keys",
        allow_none=False,
    )


def read_csv_file_into_dict(
    file_path: str | Path,
    verbose: bool = True,
    delimiter: str = ",",
) -> dict[str, list[float]]:
    type_checks.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    type_checks.ensure_char(
        param=delimiter,
        param_name="delimiter",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    if not io_manager.does_file_exist(file_path):
        raise FileNotFoundError(f"No csv-file found: {file_path}")
    if verbose:
        log_manager.log_task(f"Reading csv-file: {file_path}")
    with open(file_path, "r", newline="", encoding="utf-8") as file_pointer:
        csv_reader = csv.DictReader(
            file_pointer,
            delimiter=delimiter,
        )
        if csv_reader.fieldnames is None:
            raise ValueError("CSV is empty or missing a header row.")
        dataset: dict[str, list[float]] = {key: [] for key in csv_reader.fieldnames}
        for entry_index, entry in enumerate(csv_reader, start=2):  # header is line 1
            for key, value in entry.items():
                if (value is None) or (value == ""):
                    raise ValueError(
                        f"Missing value in column `{key}` on line {entry_index}.",
                    )
                try:
                    dataset[key].append(float(value))
                except ValueError as error:
                    raise ValueError(
                        f"Non-numeric value in column `{key}` on line {entry_index}: {value!r}.",
                    ) from error
    return dataset


def save_dict_to_csv_file(
    file_path: str | Path,
    input_dict: dict,
    overwrite: bool = True,
    verbose: bool = True,
) -> None:
    type_checks.ensure_bool(
        param=overwrite,
        param_name="overwrite",
        allow_none=False,
    )
    type_checks.ensure_bool(
        param=verbose,
        param_name="verbose",
        allow_none=False,
    )
    file_path = _ensure_path_is_valid(file_path)
    _validate_input_dict(input_dict)
    file_exists = io_manager.does_file_exist(file_path)
    if file_exists:
        if overwrite:
            _write_csv(
                file_path=file_path,
                input_dict=input_dict,
            )
            if verbose:
                log_manager.log_action(
                    title="Save CSV file",
                    outcome=log_manager.ActionOutcome.SUCCESS,
                    message="Overwrote csv-file.",
                    notes={
                        "file": str(file_path),
                        "mode": "overwrite",
                    },
                )
        else:
            _update_csv(
                file_path=file_path,
                input_dict=input_dict,
            )
            if verbose:
                log_manager.log_action(
                    title="Save CSV file",
                    outcome=log_manager.ActionOutcome.SUCCESS,
                    message="Extended csv-file.",
                    notes={
                        "file": str(file_path),
                        "mode": "extend",
                    },
                )
    else:
        _write_csv(
            file_path=file_path,
            input_dict=input_dict,
        )
        if verbose:
            log_manager.log_action(
                title="Save CSV file",
                outcome=log_manager.ActionOutcome.SUCCESS,
                message="Saved csv-file.",
                notes={
                    "file": str(file_path),
                    "mode": "create",
                },
            )


def _write_csv(
    file_path: Path,
    input_dict: dict,
) -> None:
    dataset_shape = [len(column) for column in input_dict.values()]
    if len(set(dataset_shape)) != 1:
        raise ValueError(
            "All dataset columns should be the same length."
            f" Provided `input_dict` shape: {dataset_shape}.",
        )
    with open(file_path, "w", newline="", encoding="utf-8") as file_pointer:
        writer = csv.writer(file_pointer, delimiter=",")
        writer.writerow(input_dict.keys())
        writer.writerows(zip(*input_dict.values()))


def _update_csv(
    file_path: Path,
    input_dict: dict,
) -> None:
    existing_dataset = read_csv_file_into_dict(
        file_path=file_path,
        verbose=False,
    )
    ## the following assumes each column has the same length
    existing_column_length = len(
        next(iter(existing_dataset.values())),
    )
    ## for columns that already exist, check that the amount they grow by are the same
    growth_of_existing_columns = None
    for key in input_dict:
        if key in existing_dataset:
            input_column_length = len(input_dict[key])
            if growth_of_existing_columns is None:
                growth_of_existing_columns = input_column_length
            elif input_column_length != growth_of_existing_columns:
                raise ValueError(
                    "Inconsistent append lengths for existing keys:"
                    f" expected {growth_of_existing_columns},"
                    f" got {input_column_length} for `{key}`.",
                )
    if growth_of_existing_columns is None:
        growth_of_existing_columns = 0  # no existing columns are being extended
    expected_final_column_length = existing_column_length + growth_of_existing_columns
    ## check that new columns have the right length
    for key in input_dict:
        if key not in existing_dataset:
            input_column_length = len(input_dict[key])
            if input_column_length != expected_final_column_length:
                raise ValueError(
                    "New column `{key}` must have length"
                    f" {expected_final_column_length} (existing rows + growth),"
                    f" but got {input_column_length}.",
                )
    ## apply updates
    for key in input_dict:
        if key in existing_dataset:
            existing_dataset[key].extend(input_dict[key])
        else:
            existing_dataset[key] = input_dict[key]
    ## final sanity check before saving
    final_dataset_shape = [len(column) for column in existing_dataset.values()]
    if len(set(final_dataset_shape)) != 1:
        raise ValueError(
            f"Final dataset has inconsistent column lengths: {final_dataset_shape}",
        )
    _write_csv(file_path, existing_dataset)


## } MODULE
