## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from copy import deepcopy
from Loki.Utils import Utils4Vars


## ###############################################################
## FUNCTIONS
## ###############################################################
def mergeDicts(
    dict_1: dict,
    dict_2: dict
  ) -> dict:
  """Recursively merges two dictionaries without modifying the originals."""
  Utils4Vars.assertType(dict_1, dict)
  Utils4Vars.assertType(dict_2, dict)
  dict_out = dict_1.copy()
  for key, value in dict_2.items():
    if key in dict_out:
      # Case 1: Both are dictionaries, merge them recursively
      if isinstance(dict_out[key], dict) and isinstance(value, dict):
        dict_out[key] = mergeDicts(dict_out[key], value)
      # Case 2: Both are lists, concatenate them
      elif isinstance(dict_out[key], list) and isinstance(value, list):
        dict_out[key] = dict_out[key] + value  # Concatenate lists
      # Case 3: Both are sets, union them
      elif isinstance(dict_out[key], set) and isinstance(value, set):
        dict_out[key] = dict_out[key] | value  # Union sets
      # Case 4: Other types, deepcopy to avoid modifying original dict_1
      elif isinstance(value, (dict, list, set)):
        dict_out[key] = deepcopy(value)
      # Case 5: Other values, replace directly
      else: dict_out[key] = value
    else: dict_out[key] = value
  return dict_out

def filterDict2ExcludeKeys(
    dict_in: dict,
    list_keys: list
  ) -> dict:
  Utils4Vars.assertType(dict_in, dict)
  Utils4Vars.assertType(list_keys, list)
  return {
    key : value
    for key, value in dict_in.items()
    if key not in list_keys
  }

def checkIfDictsAreDifferent(
      dict_new: dict,
      dict_ref: dict
    ) -> bool:
    ## check that the dictionaries have the same number of keys
    if len(dict_new) != len(dict_ref): return True
    ## check if any key in dict_ref is not in dict_new or if their values are different
    for key in dict_ref:
      if (key not in dict_new) or (dict_ref[key] != dict_new[key]):
        return True
    ## otherwise the dictionaries are the same
    return False

def printDict(
    input_dict: dict,
    indent : int = 0
  ):
  def _printWithIndent(indent, str_pre, str_post=None):
    if not isinstance(str_pre, str): str_pre = str(str_pre)
    if str_post is None: print(" " * indent + str_pre)
    else:                print(" " * indent + f"{str_pre} : {str_post}")
  def _shorten_and_format(value):
    if isinstance(value, (list, numpy.ndarray)):
      return list(value[:3]) + ["..."] if len(value) > 3 else value
    return value
  def _printDict(d, indent):
    for key in sorted(d.keys()):
      value = d[key]
      value_type = f"[{type(value).__name__}]"
      if isinstance(value, dict):
        _printWithIndent(indent, key, "[dict]")
        _printDict(value, indent+4)
      elif isinstance(value, numpy.ndarray):
        _printWithIndent(indent, key, value_type)
        shortened_value = _shorten_and_format(value)
        _printWithIndent(indent+4, value.shape, shortened_value)
      elif isinstance(value, list):
        _printWithIndent(indent, key)
        shortened_value = _shorten_and_format(value)
        _printWithIndent(indent+4, len(value), shortened_value)
      else:
        _printWithIndent(indent, key, f"[{value_type}]")
        _printWithIndent(indent+4, value_type, value)
  _printDict(input_dict, indent)


## END OF MODULE