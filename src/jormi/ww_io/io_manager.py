## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
import shutil
import inspect
from pathlib import Path
from typing import Union, List
from jormi.utils import list_utils


## ###############################################################
## UTILITY FUNCTIONS
## ###############################################################

def get_caller_directory() -> Path:
  """Get the directory of the script that invoked this function."""
  caller_frame = inspect.stack()[1]
  caller_file_path = caller_frame.filename
  return Path(caller_file_path).resolve().parent

def combine_file_path_parts(file_path_parts : list[str | Path]) -> Path:
  return Path(*list_utils.flatten_list(list_utils.filter_out_nones(file_path_parts))).absolute()

def resolve_file_path(
    file_path : str | Path | None = None,
    directory : str | Path | None = None,
    file_name : str | None = None,
  ):
  if file_path is None:
    missing = []
    if (directory is None): missing.append("directory")
    if (file_name is None): missing.append("file_name")
    if missing:
      raise ValueError(
        "You have not provided enough information about the file and where it is. "
        f"You are missing: {list_utils.cast_to_string(missing)}. "
        "Alternatively, provide `file_path` directly."
      )
    file_path = combine_file_path_parts(list_utils.filter_out_nones([ directory, file_name ]))
  else: file_path = Path(file_path).absolute()
  return file_path

def does_directory_exist(
    directory   : str | Path,
    raise_error : bool = False,
  ) -> bool:
  directory = Path(directory).absolute()
  result = directory.is_dir()
  if not(result) and raise_error: raise NotADirectoryError(f"Directory does not exist: {directory}")
  return result

def init_directory(
    directory : str | Path,
    verbose   : bool = True,
  ):
  directory = Path(directory).resolve(strict=False)
  if not does_directory_exist(directory):
    directory.mkdir(parents=True)
    if verbose: print("Initialised directory:", directory)
  elif verbose: print("Directory already exists:", directory)

def does_file_exist(
    file_path   : str | Path | None = None,
    directory   : str | Path | None = None,
    file_name   : str | None = None,
    raise_error : bool = False,
  ) -> bool:
  file_path = resolve_file_path(file_path=file_path, directory=directory, file_name=file_name)
  file_path_exists = file_path.is_file()
  if not(file_path_exists) and raise_error:
    raise FileNotFoundError(f"File does not exist: {file_path}")
  return file_path_exists

def _resolve_and_validate_file_operation(
    directory_from : str | Path,
    directory_to   : str | Path,
    file_name      : str,
    overwrite      : bool = False,
    dry_run        : bool = False,
  ) -> tuple[Path, Path]:
  does_directory_exist(directory=directory_from, raise_error=True)
  file_path_from = combine_file_path_parts([ directory_from, file_name ])
  does_file_exist(file_path=file_path_from, raise_error=True)
  if not does_directory_exist(directory=directory_to):
    if not dry_run:
      init_directory(directory=directory_to, verbose=False)
    else: print(f"Would create directory: {directory_to}")
  file_path_to = combine_file_path_parts([ directory_to, file_name ])
  if not(overwrite) and does_file_exist(file_path=file_path_to, raise_error=False):
    raise FileExistsError(f"File already exists: {file_path_to}")
  return file_path_from, file_path_to

def _print_file_action(
    action         : str,
    file_name      : str,
    directory_from : str | Path,
    directory_to   : str | Path | None = None,
  ):
  print(f"{action}:")
  print(f"\t> File: {file_name}")
  print(f"\t> From: {directory_from}")
  if directory_to is not None:
    print(f"\t> To:   {directory_to}")

def copy_file(
    directory_from : str | Path,
    directory_to   : str | Path,
    file_name      : str,
    overwrite      : bool = False,
    dry_run        : bool = False,
    verbose        : bool = True,
  ):
  file_path_from, file_path_to = _resolve_and_validate_file_operation(
    directory_from = directory_from,
    directory_to   = directory_to,
    file_name      = file_name,
    overwrite      = overwrite,
    dry_run        = dry_run
  )
  if not dry_run:
    shutil.copy(file_path_from, file_path_to)
    shutil.copymode(file_path_from, file_path_to)
  if verbose or dry_run:
    _print_file_action(
      action         = "Copied" if not dry_run else "Would copy",
      file_name      = file_name,
      directory_from = directory_from,
      directory_to   = directory_to
    )

def move_file(
    directory_from : str | Path,
    directory_to   : str | Path,
    file_name      : str,
    overwrite      : bool = False,
    dry_run        : bool = False,
    verbose        : bool = True,
  ):
  file_path_from, file_path_to = _resolve_and_validate_file_operation(
    directory_from = directory_from,
    directory_to   = directory_to,
    file_name      = file_name,
    overwrite      = overwrite,
    dry_run        = dry_run
  )
  if not dry_run: shutil.move(file_path_from, file_path_to)
  if verbose or dry_run:
    _print_file_action(
      action         = "Moved" if not dry_run else "Would move",
      file_name      = file_name,
      directory_from = directory_from,
      directory_to   = directory_to
    )

def delete_file(
    directory : str | Path,
    file_name : str,
    dry_run   : bool = False,
    verbose   : bool = True,
  ):
  does_directory_exist(directory=directory, raise_error=True)
  file_path = combine_file_path_parts([ directory, file_name ])
  if not dry_run: file_path.unlink()
  if verbose or dry_run:
    _print_file_action(
      action         = "Deleted" if not dry_run else "Would delete",
      file_name      = file_name,
      directory_from = directory
    )

class ItemFilter:
  def __init__(
      self,
      *,
      include_string  : Union[str, List[str], None] = None,
      exclude_string  : Union[str, List[str], None] = None,
      prefix          : str | None = None,
      suffix          : str | None = None,
      delimiter       : str = "_",
      num_parts       : int | None = None,
      index_of_value  : int | None = None,
      min_value       : int | float = 0,
      max_value       : int | float = numpy.inf,
      include_files   : bool = True,
      include_folders : bool = True
    ):
    self.include_string  = self._to_list(include_string)
    self.exclude_string  = self._to_list(exclude_string)
    self.prefix          = prefix
    self.suffix          = suffix
    self.delimiter       = delimiter
    self.num_parts       = num_parts
    self.index_of_value  = index_of_value
    self.min_value       = min_value
    self.max_value       = max_value
    self.include_files   = include_files
    self.include_folders = include_folders
    self._validate_inputs()

  def _to_list(self, value):
    if value is None: return []
    if isinstance(value, str): return [value]
    if isinstance(value, list): return value
    raise ValueError("Expected a string or list of strings.")

  def _validate_inputs(self):
    if not (self.include_files or self.include_folders):
      raise ValueError("At least one of `include_files` or `include_folders` must be enabled.")
    if not isinstance(self.min_value, (int, float)) or not isinstance(self.max_value, (int, float)):
      raise TypeError("`min_value` and `max_value` must be numbers.")
    if self.min_value > self.max_value:
      raise ValueError("`min_value` cannot be greater than `max_value`.")

  def _meets_criteria(self, item_path: Path) -> bool:
    if item_path.is_file() and not self.include_files: return False
    if item_path.is_dir() and not self.include_folders: return False
    item_name = item_path.name
    if self.include_string and not all(
      include_string in item_name
      for include_string in self.include_string
    ): return False
    if self.exclude_string and any(
      exclude_string in item_name
      for exclude_string in self.exclude_string
    ): return False
    if self.prefix and not item_name.startswith(self.prefix): return False
    if self.suffix and not item_name.endswith(self.suffix): return False
    name_parts = item_name.split(self.delimiter)
    if (self.num_parts is not None) and (len(name_parts) != self.num_parts): return False
    if self.index_of_value is not None:
      if len(name_parts) < abs(self.index_of_value): return False
      try: value = int(name_parts[self.index_of_value])
      except ValueError: return False
      if not (self.min_value <= value <= self.max_value): return False
    return True

  def filter(self, directory: Union[str, Path]) -> List[Path]:
    """Filter item names in the given directory based on current criteria."""
    directory = Path(directory).absolute()
    does_directory_exist(directory, raise_error=True)
    return sorted(
      item for item in directory.iterdir()
      if self._meets_criteria(item)
    )


## END OF MODULE