## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def getEveryNthElement(list_elems, num_elems):
  if num_elems > len(list_elems): return list_elems
  index_step = (len(list_elems) - 1) // (num_elems - 1)
  return list_elems[::index_step][:num_elems]

def getIntersectOfLists(list1, list2):
  return sorted(set(list1) & set(list2))

def getUnionOfLists(list1, list2):
  return sorted(set(list1) | set(list2))

def getIndexOfClosestValue(input_vals, target_value):
  array_vals = numpy.asarray(input_vals)
  ## check there are sufficient points
  if array_vals.shape[0] < 3: raise Exception(f"Error: There is an insuffient number of elements in:", input_vals)
  if target_value is None:    return None
  if target_value ==  numpy.inf: return numpy.nanargmax(array_vals)
  if target_value == -numpy.inf: return numpy.nanargmin(array_vals)
  return numpy.nanargmin(numpy.abs(array_vals - target_value))

def extendListToMatchLength(list_input, list_ref):
  if len(list_input) < len(list_ref):
    list_input.extend(
      [ list_input[0] ] * int( len(list_ref) - len(list_input) )
    )

def flattenList(list_elems):
  return list(numpy.concatenate(list_elems).flat)


## END OF MODULE