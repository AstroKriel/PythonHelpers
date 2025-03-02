## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os
import re
import shutil
import subprocess
import numpy as np

## load user defined modules
from Loki.TheUsefulModule import WWTerminal, WWLists


## ###############################################################
## INTERACTING WITH FILES AND FOLDERS
## ###############################################################
def createFilepathString(list_directories_and_filename):
  return os.path.normpath(
    os.path.join(
      *WWLists.flattenList(list_directories_and_filename)
    )
  )

def checkIfFileExists(directory, filename, bool_trigger_error=False):
  filepath_file = createFilepathString([directory, filename])
  bool_filepath_exists = os.path.isfile(filepath_file)
  if not(bool_filepath_exists) and bool_trigger_error:
    raise Exception(f"Error: File does not exist: {filepath_file}")
  else: return bool_filepath_exists

def checkIfDirectoryExists(directory):
  return os.path.isdir(directory)

def initDirectory(directory, bool_verbose=True):
  if not(checkIfDirectoryExists(directory)):
    os.makedirs(directory)
    if bool_verbose: print("Successfully initialised directory:", directory)
  elif bool_verbose: print("No need to initialise diectory (already exists):", directory)


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
    last_value     = np.inf,
  ):
  """
    Create a filter function for filenames based on various conditions.
    
    Parameters:
    - contains       : Filter filenames that contain this string.
    - not_contains   : Filter filenames that do not contain this string.
    - starts_with    : Filter filenames that start with this string.
    - ends_with      : Filter filenames that end with this string.
    - split_by       : The delimiter to split the filename into parts (default: "_").
    - num_parts      : Filter filenames with this number of parts when split by `split_by`.
    - value_location : The index of the part to check for a range of values.
    - first_value    : The minimum value (inclusive) for the `value_location` part.
    - last_value     : The maximum value (inclusive) for the `value_location` part.
    
    Returns:
    - A filter function that takes a filename and returns `True` if it matches the criteria, else `False`.
  """
  def meetsCondition(filename):
    list_filename_parts = filename.split(split_by)
    ## if basic conditions are met then proceed
    if not all([
        (contains     is None) or (contains in filename),
        (not_contains is None) or not(not_contains in filename),
        (starts_with  is None) or filename.startswith(starts_with),
        (ends_with    is None) or filename.endswith(ends_with),
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
    last_value     = np.inf,
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
    lambda filename: obj_filter(filename) and checkIfFileExists(directory, filename),
    list_filenames
  )
  return list(list_filenames_filtered)


## ###############################################################
## WORKING WITH FILES
## ###############################################################
def copyFile(directory_from, directory_to, filename, bool_overwrite=False, bool_verbose=True):
  filepath_file_from = createFilepathString([ directory_from, filename ])
  filepath_file_to   = createFilepathString([ directory_to, filename ])
  if not checkIfDirectoryExists(directory_from):
    raise NotADirectoryError(f"Error: Source directory does not exist: {directory_from}")
  if not checkIfDirectoryExists(directory_to):
    initDirectory(directory_to, bool_verbose)
  checkIfFileExists(directory_from, filename, bool_trigger_error=True)
  if not(bool_overwrite) and checkIfFileExists(directory_to, filename, bool_trigger_error=False):
    raise FileExistsError(f"Error: File already exists: {filepath_file_to}")
  ## copy the file and it`s permissions
  shutil.copy(filepath_file_from, filepath_file_to)
  shutil.copymode(filepath_file_from, filepath_file_to)
  if bool_verbose:
    print(f"Coppied:")
    print(f"\t> File: {filename}")
    print(f"\t> From: {directory_from}")
    print(f"\t> To:   {directory_to}")


## END OF LIBRARY