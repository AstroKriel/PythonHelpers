## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def computeNorm(array1, array2, p=2, bool_normalise=False):
  """Compute the p-norm between two arrays and optionally normalise by num_points^(1/p)."""
  array1, array2 = numpy.asarray(array1), numpy.asarray(array2)
  if array1.shape != array2.shape: raise ValueError("Arrays must have the same shape")
  if len(array1) == 0: raise ValueError("Arrays cannot be empty.")
  if not isinstance(p, (int, float)): raise ValueError(f"Invalid norm order `p={p}`. Must be a number.")
  if numpy.all(array1 == array2): return 0
  array_diff = numpy.abs(array1 - array2)
  if p == numpy.inf: return numpy.max(array_diff)
  elif p == 1:
    ## L1 norm: sum of absolute differences
    result = numpy.sum(array_diff)
    if bool_normalise: result /= len(array1)
  elif p == 0:
    ## L0 pseudo-norm: count of non-zero elements
    result = numpy.count_nonzero(array_diff)
  elif p > 0:
    ## general case for p > 0
    ## note numerical stability improvement: scale by maximum value
    max_diff = numpy.max(array_diff)
    if max_diff > 0:
        scaled_diff = array_diff / max_diff
        result = max_diff * numpy.power(numpy.sum(numpy.power(scaled_diff, p)), 1/p)
        if bool_normalise: result /= numpy.power(len(array1), 1/p)
    else: result = 0
  else: raise ValueError(f"Invalid norm order `p={p}`. Must be positive or infinity.")
  return result


## END OF MODULE