## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import shutil
import numpy
from Loki.Utils import Utils4Lists


## ###############################################################
## INTERACTING WITH FILES AND FOLDERS
## ###############################################################
def createFilepathString(list_directories_and_filename):
  return os.path.normpath(
    os.path.join(
      *Utils4Lists.flattenList(list_directories_and_filename)
    )
  )

def checkIfDirectoryExists(directory):
  return os.path.isdir(directory)

def initDirectory(directory, bool_verbose=True):
  if not(checkIfDirectoryExists(directory)):
    os.makedirs(directory)
    if bool_verbose: print("Successfully initialised directory:", directory)
  elif bool_verbose: print("No need to initialise diectory (already exists):", directory)

def checkIfFileExists(directory, file_name, bool_trigger_error=False):
  file_path = createFilepathString([directory, file_name])
  bool_file_path_exists = os.path.isfile(file_path)
  if not(bool_file_path_exists) and bool_trigger_error:
    raise Exception(f"Error: File does not exist: {file_path}")
  else: return bool_file_path_exists

def copyFile(directory_from, directory_to, file_name, bool_overwrite=False, bool_verbose=True):
  file_path_from = createFilepathString([ directory_from, file_name ])
  file_path_to   = createFilepathString([ directory_to, file_name ])
  if not checkIfDirectoryExists(directory_from):
    raise NotADirectoryError(f"Error: Source directory does not exist: {directory_from}")
  if not checkIfDirectoryExists(directory_to):
    initDirectory(directory_to, bool_verbose)
  checkIfFileExists(directory_from, file_name, bool_trigger_error=True)
  if not(bool_overwrite) and checkIfFileExists(directory_to, file_name, bool_trigger_error=False):
    raise FileExistsError(f"Error: File already exists: {file_path_to}")
  ## copy the file and it`s permissions
  shutil.copy(file_path_from, file_path_to)
  shutil.copymode(file_path_from, file_path_to)
  if bool_verbose:
    print(f"Coppied:")
    print(f"\t> File: {file_name}")
    print(f"\t> From: {directory_from}")
    print(f"\t> To:   {directory_to}")


## ###############################################################
## FILTERING FILES IN DIRECTORY
## ###############################################################
def makeFilter(
    contains       = None,
    not_contains   = None,
    starts_with    = None,
    ends_with      = None,
    split_by       = "_",
    num_parts      = None,
    value_location = None,
    first_value    = 0,
    last_value     = numpy.inf,
  ):
  """
    Create a filter function for file names based on various conditions.
    
    Parameters:
    - contains       : Filter file names that contain this string.
    - not_contains   : Filter file names that do not contain this string.
    - starts_with    : Filter file names that start with this string.
    - ends_with      : Filter file names that end with this string.
    - split_by       : The delimiter to split the file_name into parts (default: "_").
    - num_parts      : Filter file names with this number of parts when split by `split_by`.
    - value_location : The index of the part to check for a range of values.
    - first_value    : The minimum value (inclusive) for the `value_location` part.
    - last_value     : The maximum value (inclusive) for the `value_location` part.
    
    Returns:
    - A filter function that takes a file_name and returns `True` if it matches the criteria, else `False`.
  """
  def meetsCondition(file_name):
    list_filename_parts = file_name.split(split_by)
    ## if basic conditions are met then proceed
    if not all([
        (contains     is None) or (contains in file_name),
        (not_contains is None) or not(not_contains in file_name),
        (starts_with  is None) or file_name.startswith(starts_with),
        (ends_with    is None) or file_name.endswith(ends_with),
        (num_parts    is None) or (len(list_filename_parts) == num_parts),
      ]): return False
    if value_location is not None:
      ## check that the value falls within the specified range
      if len(list_filename_parts) > abs(value_location):
        value = int(list_filename_parts[value_location])
        return (first_value <= value) and (value <= last_value)
    return True
  return meetsCondition

def getFilesInDirectory(
    directory,
    contains       = None,
    not_contains   = None,
    starts_with    = None,
    ends_with      = None,
    split_by       = "_",
    num_parts      = None,
    value_location = None,
    first_value    = 0,
    last_value     = numpy.inf,
  ):
  obj_filter = makeFilter(
    contains       = contains,
    not_contains   = not_contains,
    starts_with    = starts_with,
    ends_with      = ends_with,
    split_by       = split_by,
    num_parts      = num_parts,
    value_location = value_location,
    first_value    = first_value,
    last_value     = last_value,
  )
  list_filenames = os.listdir(directory)
  list_filenames_filtered = filter(
    lambda file_name: obj_filter(file_name) and checkIfFileExists(directory, file_name),
    list_filenames
  )
  return list(list_filenames_filtered)


## END OF MODULE