## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import json
import copy
import numpy
from Loki.Utils import Utils4Dicts


## ###############################################################
## FUNCTIONS
## ###############################################################
def readJsonFile2Dict(directory, filename, bool_verbose=True):
  filepath_file = f"{directory}/{filename}"
  if os.path.isfile(filepath_file):
    if bool_verbose: print("Reading in json-file:", filepath_file)
    with open(filepath_file, "r") as fp:
      return copy.deepcopy(json.load(fp))
  else: raise Exception(f"Error: No json-file found: {filepath_file}")

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if   isinstance(obj, numpy.integer):  return int(obj)
    elif isinstance(obj, numpy.floating): return float(obj)
    elif isinstance(obj, numpy.bool_):    return bool(obj)
    elif isinstance(obj, numpy.ndarray):  return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def saveObj2JsonFile(filepath_file, obj, bool_verbose=True):
  with open(filepath_file, "w") as fp:
    json.dump(
      obj       = vars(obj), # store member-variables in a dictionary
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Saved json-file:", filepath_file)

def saveDict2JsonFile(filepath_file, input_dict, bool_verbose=True):
  if os.path.isfile(filepath_file): appendDict2JsonFile(filepath_file, input_dict, bool_verbose)
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
  with open(filepath_file, "r") as fp_r:
    dict_old = json.load(fp_r)
  Utils4Dicts.mergeDicts(dict_old, dict2add)
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