## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def mergeDicts(dict_ref, dict2add):
  for key, value in dict2add.items():
    if key in dict_ref and isinstance(dict_ref[key], dict) and isinstance(value, dict):
      mergeDicts(dict_ref[key], value)
    else: dict_ref[key] = value

def filterDict2ExcludeKeys(input_dict, list_keys):
  return {
    k : v
    for k, v in input_dict.items()
    if k not in list_keys
  }

def checkIfDictsAreDifferent(dict_new, dict_ref):
    ## check that the dictionaries have the same number of keys
    if len(dict_new) != len(dict_ref): return True
    ## check if any key in dict_ref is not in dict_new or if their values are different
    for key in dict_ref:
      if (key not in dict_new) or (dict_ref[key] != dict_new[key]):
        return True
    ## otherwise the dictionaries are the same
    return False

def printDict(input_dict, indent=0):
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