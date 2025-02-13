## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, json, copy, h5py
import numpy as np


## ###############################################################
## WORKING WITH DICTIONARIES
## ###############################################################
def mergeDicts(dict_ref, dict2add):
  ## recursively merge dict2add into dict_ref
  for key, value in dict2add.items():
    if key in dict_ref and isinstance(dict_ref[key], dict) and isinstance(value, dict):
      mergeDicts(dict_ref[key], value)
    else: dict_ref[key] = value

def getDictWithoutKeys(input_dict, list_keys):
  return {
    k : v
    for k, v in input_dict.items()
    if k not in list_keys
  }

def areDictsDifferent(dict_new, dict_ref):
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
    else:                print(" " * indent + f"'{str_pre}' : {str_post}")
  def _shorten_and_format(value):
    if isinstance(value, (list, np.ndarray)):
      return list(value[:3]) + ["..."] if len(value) > 3 else value
    return value
  def _printDict(d, indent):
    for key in sorted(d.keys()):
      value = d[key]
      value_type = f"[{type(value).__name__}]"
      if isinstance(value, dict):
        _printWithIndent(indent, key, "[dict]")
        _printDict(value, indent+4)
      elif isinstance(value, np.ndarray):
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


## ###############################################################
## WORKING WITH JSON-FILES
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
    if   isinstance(obj, np.integer):  return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.bool_):    return bool(obj)
    elif isinstance(obj, np.ndarray):  return obj.tolist()
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
  if os.path.isfile(filepath_file):
    appendDict2JsonFile(filepath_file, input_dict, bool_verbose)
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
  mergeDicts(dict_old, dict2add)
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


## ###############################################################
## WORKING WITH OBJECTS
## ###############################################################
def updateObjAttr(obj, attr, desired_val):
  ## check that the new attribute value is not None
  if desired_val is not None:
    ## check that the new value is not the same as the old value
    if not(getattr(obj, attr) == desired_val):
      ## change the attribute value
      setattr(obj, attr, desired_val)
      return True
  ## don't change the attribute value
  return False

def printObjAttrNames(obj):
  ## loop over all the attribute-names in the object
  for attr in vars(obj):
    print(attr)


## ###############################################################
## WORKING WITH HDF5 FILES
## ###############################################################
def deleteEmptyGroupsHDF5(filepath_file):
  ## helper function
  def findEmptyGroups(group):
    for _, item in group.items():
      if isinstance(item, h5py.Group):
        if len(item) == 0: groups_to_delete.append(item.name)
        else:              findEmptyGroups(item)
  ## do stuff
  with h5py.File(filepath_file, 'a') as hdf:
    groups_to_delete = []
    findEmptyGroups(hdf)
    for group_name in groups_to_delete:
      del hdf[group_name]
  repackHDF5(filepath_file)

def repackHDF5(filepath_file):
  ## helper function
  def _recursive_copy(source, destination):
    for name, item in source.items():
      if isinstance(item, h5py.Group):
        new_group = destination.create_group(name)
        _recursive_copy(item, new_group)
      elif isinstance(item, h5py.Dataset):
        destination.create_dataset(name, data=item[()])
  ## do stuff
  _filepath_file = filepath_file + '_temp'
  with h5py.File(filepath_file, 'r') as old_file:
    with h5py.File(_filepath_file, 'w') as new_file:
      _recursive_copy(old_file, new_file)
  os.remove(filepath_file)
  os.rename(_filepath_file, filepath_file)
  print("Repacked:", filepath_file)


## END OF LIBRARY