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
def readJsonFile2Dict(directory, file_name, bool_verbose=True):
  file_path = f"{directory}/{file_name}"
  if os.path.isfile(file_path):
    if bool_verbose: print("Reading in json-file:", file_path)
    with open(file_path, "r") as fp:
      return copy.deepcopy(json.load(fp))
  else: raise Exception(f"Error: No json-file found: {file_path}")

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if   isinstance(obj, numpy.integer):  return int(obj)
    elif isinstance(obj, numpy.floating): return float(obj)
    elif isinstance(obj, numpy.bool_):    return bool(obj)
    elif isinstance(obj, numpy.ndarray):  return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def saveObj2JsonFile(file_path, obj, bool_verbose=True):
  with open(file_path, "w") as fp:
    json.dump(
      obj       = vars(obj), # store member-variables in a dictionary
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Saved json-file:", file_path)

def saveDict2JsonFile(file_path, input_dict, bool_verbose=True):
  if os.path.isfile(file_path): appendDict2JsonFile(file_path, input_dict, bool_verbose)
  else: createJsonFile(file_path, input_dict, bool_verbose)

def createJsonFile(file_path, dict2save, bool_verbose=True):
  file_path = file_path.replace("//", "/")
  with open(file_path, "w") as fp:
    json.dump(
      obj       = dict2save,
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Saved json-file:", file_path)

def appendDict2JsonFile(file_path, dict2add, bool_verbose=True):
  with open(file_path, "r") as fp_r:
    dict_old = json.load(fp_r)
  Utils4Dicts.mergeDicts(dict_old, dict2add)
  with open(file_path, "w+") as fp_w:
    json.dump(
      obj       = dict_old,
      fp        = fp_w,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if bool_verbose: print("Updated json-file:", file_path)


## END OF MODULE