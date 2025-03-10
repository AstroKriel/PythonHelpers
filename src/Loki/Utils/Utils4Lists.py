## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.Utils import Utils4Vars


## ###############################################################
## FUNCTIONS
## ###############################################################
def getIntersectOfLists(
    list1: list,
    list2: list,
    bool_sort: bool=False
  ) -> list:
  """Returns the intersection of two lists, optionally sorted."""
  Utils4Vars.assertType(list1, list)
  Utils4Vars.assertType(list2, list)
  if (len(list1) == 0) or (len(list2) == 0): return []
  set_intersect = set(list1) & set(list2)
  return sorted(set_intersect) if bool_sort else list(set_intersect)

def getUnionOfLists(
    list1: list,
    list2: list,
    bool_sort: bool=False
  ) -> list:
  """Returns the union of two lists, optionally sorted."""
  Utils4Vars.assertType(list1, list)
  Utils4Vars.assertType(list2, list)
  if (len(list1) == 0) or (len(list2) == 0): return list1 + list2
  set_union = set(list1) | set(list2)
  return sorted(set_union) if bool_sort else list(set_union)

def getIndexOfClosestValue(
    list_vals: list,
    target_value: float
  ) -> int:
  """Finds the index of the value in `list_vals` that is closest to `target_value`."""
  Utils4Vars.assertType(list_vals, list)
  Utils4Vars.assertType(target_value, (int, float))
  if len(list_vals) == 0: raise ValueError("Input list cannot be empty")
  array_vals = numpy.asarray(list_vals)
  if target_value is None: return None
  if target_value ==  numpy.inf: return int(numpy.nanargmax(array_vals))
  if target_value == -numpy.inf: return int(numpy.nanargmin(array_vals))
  return int(numpy.nanargmin(numpy.abs(array_vals - target_value)))

def flattenList(list_elems: list) -> list:
  """Flattens a nested list into a single list."""
  Utils4Vars.assertType(list_elems, list)
  if all(
    not isinstance(elem, list)
    for elem in list_elems
  ): return list_elems
  return list(numpy.concatenate(list_elems).flat)


## END OF MODULE