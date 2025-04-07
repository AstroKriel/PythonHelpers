## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from loki.utils import var_utils


## ###############################################################
## FUNCTIONS
## ###############################################################
def get_intersect_of_lists(
    list_a: list,
    list_b: list,
    sort_values: bool = False,
  ) -> list:
  """Find the intersection of two lists (optionally sorted)."""
  var_utils.assert_type(list_a, list)
  var_utils.assert_type(list_b, list)
  if (len(list_a) == 0) or (len(list_b) == 0): return []
  set_intersect = set(list_a) & set(list_b)
  return sorted(set_intersect) if sort_values else list(set_intersect)

def get_union_of_lists(
    list_a: list,
    list_b: list,
    sort_values: bool = False,
  ) -> list:
  """Find the union of two lists (optionally sorted)."""
  var_utils.assert_type(list_a, list)
  var_utils.assert_type(list_b, list)
  if (len(list_a) == 0) or (len(list_b) == 0): return list_a + list_b
  set_union = set(list_a) | set(list_b)
  return sorted(set_union) if sort_values else list(set_union)

def get_index_of_closest_value(
    values: list,
    target: float,
  ) -> int:
  """Find the index of the closest value to a `target` value."""
  var_utils.assert_type(values, list)
  var_utils.assert_type(target, (int, float))
  if len(values) == 0: raise ValueError("Input list cannot be empty")
  array_vals = numpy.asarray(values)
  if target is None: return None
  if target ==  numpy.inf: return int(numpy.nanargmax(array_vals))
  if target == -numpy.inf: return int(numpy.nanargmin(array_vals))
  return int(numpy.nanargmin(numpy.abs(array_vals - target)))

def flatten_list(elems: list) -> list:
  """Flatten a nested list into a single list."""
  var_utils.assert_type(elems, list)
  if all(
    not isinstance(elem, list)
    for elem in elems
  ): return elems
  return list(numpy.concatenate(elems).flat)


## END OF MODULE