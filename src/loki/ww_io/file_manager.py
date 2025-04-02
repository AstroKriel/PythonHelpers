## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import shutil
import numpy
from loki.utils import list_utils


## ###############################################################
## INTERACTING WITH FILES AND FOLDERS
## ###############################################################
def create_file_path(file_path_elems):
  return os.path.normpath(
    os.path.join(
      *list_utils.flatten_list(file_path_elems)
    )
  )

def does_directory_exist(directory):
  return os.path.isdir(directory)

def init_directory(directory, verbose=True):
  if not does_directory_exist(directory):
    os.makedirs(directory)
    if verbose: print("Successfully initialised directory:", directory)
  elif verbose: print("No need to initialise diectory (already exists):", directory)

def does_file_exist(directory, file_name, raise_error_if_not_found=False):
  file_path = create_file_path([directory, file_name])
  bool_file_path_exists = os.path.isfile(file_path)
  if not(bool_file_path_exists) and raise_error_if_not_found:
    raise Exception(f"Error: File does not exist: {file_path}")
  else: return bool_file_path_exists

def copy_file(directory_from, directory_to, file_name, bool_overwrite=False, verbose=True):
  file_path_from = create_file_path([ directory_from, file_name ])
  file_path_to   = create_file_path([ directory_to, file_name ])
  if not does_directory_exist(directory_from):
    raise NotADirectoryError(f"Error: Source directory does not exist: {directory_from}")
  if not does_directory_exist(directory_to):
    init_directory(directory_to, verbose)
  does_file_exist(directory_from, file_name, raise_error_if_not_found=True)
  if not(bool_overwrite) and does_file_exist(directory_to, file_name, raise_error_if_not_found=False):
    raise FileExistsError(f"Error: File already exists: {file_path_to}")
  ## copy the file and it`s permissions
  shutil.copy(file_path_from, file_path_to)
  shutil.copymode(file_path_from, file_path_to)
  if verbose:
    print(f"Coppied:")
    print(f"\t> File: {file_name}")
    print(f"\t> From: {directory_from}")
    print(f"\t> To:   {directory_to}")


## ###############################################################
## FILTERING FILES IN DIRECTORY
## ###############################################################
def create_filter_for_files(
    include_string,
    exclude_string,
    prefix,
    suffix,
    delimiter,
    num_parts,
    index_of_value,
    min_value,
    max_value,
  ):
  def _does_file_meet_criteria(file_name):
    list_filename_parts = file_name.split(delimiter)
    ## if basic conditions are met then proceed
    if not all([
        (include_string     is None) or (include_string in file_name),
        (exclude_string is None) or not(exclude_string in file_name),
        (prefix  is None) or file_name.startswith(prefix),
        (suffix    is None) or file_name.endswith(suffix),
        (num_parts    is None) or (len(list_filename_parts) == num_parts),
      ]): return False
    if index_of_value is not None:
      ## check that the value falls within the specified range
      if len(list_filename_parts) > abs(index_of_value):
        value = int(list_filename_parts[index_of_value])
        return (min_value <= value) and (value <= max_value)
    return True
  return _does_file_meet_criteria

def filter_files_in_directory(
    directory      : str,
    include_string : str = None,
    exclude_string : str = None,
    prefix         : str = None,
    suffix         : str = None,
    delimiter      : str = "_",
    num_parts      : int= None,
    index_of_value : int= None,
    min_value      : int = 0,
    max_value      : int = numpy.inf,
  ):
  """
    Filter file names in a `directory` based on various conditions:
    - `include_string` : File names must contain this string.
    - `exclude_string` : File names should not contain this string.
    - `prefix`         : File names should start with this string.
    - `suffix`         : File names should end with this string.
    - `delimiter`      : The delimiter used to split the file name (default: "_").
    - `num_parts`      : Only include files that, when split by `delimiter`, have this number of parts.
    - `index_of_value` : The part-index to check for value range conditions.
    - `min_value`      : The minimum valid value (inclusive) stored at `index_of_value`.
    - `max_value`      : The maximum valid value (inclusive) stored at `index_of_value`.
  """
  file_filter = create_filter_for_files(
    include_string = include_string,
    exclude_string = exclude_string,
    prefix         = prefix,
    suffix         = suffix,
    delimiter      = delimiter,
    num_parts      = num_parts,
    index_of_value = index_of_value,
    min_value      = min_value,
    max_value      = max_value,
  )
  file_names = os.listdir(directory)
  return list(filter(file_filter, file_names))


## END OF MODULE