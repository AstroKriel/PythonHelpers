## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import json
import copy
import numpy
from Loki.WWCollections import DictUtils


## ###############################################################
## FUNCTIONS
## ###############################################################
def readJsonFile2Dict(directory, filename, bool_verbose=True):
  filepath_file = f"{directory}/{filename}"
  ## read file if it exists
  if os.path.isfile(filepath_file):
    if bool_verbose: print("Reading in json-file:", filepath_file)
    with open(filepath_file, "r") as fp:
      return copy.deepcopy(json.load(fp))
  ## indicate the file was not found
  else: raise Exception(f"Error: No json-file found: {filepath_file}")

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if   isinstance(obj, numpy.integer):  return int(obj)
    elif isinstance(obj, numpy.floating): return float(obj)
    elif isinstance(obj, numpy.bool_):    return bool(obj)
    elif isinstance(obj, numpy.ndarray):  return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def saveObj2JsonFile(filepath_file, obj, bool_verbose=True):
  ## save object to file
  with open(filepath_file, "w") as fp:
    json.dump(
      obj       = vars(obj), # store obj member-variables in a dictionary
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  ## indicate success
  if bool_verbose: print("Saved json-file:", filepath_file)

def saveDict2JsonFile(filepath_file, input_dict, bool_verbose=True):
  ## if json-file already exists, then append dictionary
  if os.path.isfile(filepath_file): appendDict2JsonFile(filepath_file, input_dict, bool_verbose)
  ## create json-file with dictionary
  else: createJsonFile(filepath_file, input_dict, bool_verbose)

def createJsonFile(filepath_file, dict2save, bool_verbose=True):
  filepath_file = filepath_file.replace("//", "/")
  with open(filepath_file, "w") as fp:
    json.dump(
      obj       = dict2save,
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Saved json-file:", filepath_file)

def appendDict2JsonFile(filepath_file, dict2add, bool_verbose=True):
  ## read json-file into dict
  with open(filepath_file, "r") as fp_r:
    dict_old = json.load(fp_r)
  ## append extra contents to dict
  DictUtils.mergeDicts(dict_old, dict2add)
  ## update (overwrite) json-file
  with open(filepath_file, "w+") as fp_w:
    json.dump(
      obj       = dict_old,
      fp        = fp_w,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Updated json-file:", filepath_file)


## END OF MODULE