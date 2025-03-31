## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import json
import copy
import numpy
from Loki.Utils import Utils4Dicts, Utils4IO


## ###############################################################
## FUNCTIONS
## ###############################################################
def read_json_file_into_dict(
    directory : str,
    file_name : str,
    verbose   : bool = True
  ):
  file_path = Utils4IO.create_file_path([directory, file_name])
  if os.path.isfile(file_path):
    if verbose: print("Reading in json-file:", file_path)
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

def save_dict_to_json_file(
    file_path  : str,
    input_dict : dict,
    verbose    : bool =True
  ):
  if os.path.isfile(file_path): add_dict_to_json_file(file_path, input_dict, verbose)
  else: create_json_file_from_dict(file_path, input_dict, verbose)

def create_json_file_from_dict(
    file_path  : str,
    input_dict : dict,
    verbose    : bool =True
  ):
  file_path = file_path.replace("//", "/")
  with open(file_path, "w") as fp:
    json.dump(
      obj       = input_dict,
      fp        = fp,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if verbose: print("Saved json-file:", file_path)

def add_dict_to_json_file(
    file_path  : str,
    input_dict : dict,
    verbose    : bool = True
  ):
  with open(file_path, "r") as fp_r:
    dict_old = json.load(fp_r)
  Utils4Dicts.merge_dicts(dict_old, input_dict)
  with open(file_path, "w+") as fp_w:
    json.dump(
      obj       = dict_old,
      fp        = fp_w,
      cls       = NumpyEncoder,
      sort_keys = True,
      indent    = 2
    )
  if verbose: print("Updated json-file:", file_path)


## END OF MODULE