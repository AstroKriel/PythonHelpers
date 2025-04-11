## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
import numpy
from jormungandr.utils import list_utils


## ###############################################################
## TESTS
## ###############################################################
class TestListUtils(unittest.TestCase):

  def test_cast_to_string(self):
    result = list_utils.cast_to_string(["a", "b", "c"])
    self.assertEqual(result, "`a`, `b`, or `c`")
    ## with conjunction but no Oxford comma
    result = list_utils.cast_to_string(["a", "b", "c"], conjunction="and", use_oxford_comma=False)
    self.assertEqual(result, "`a`, `b` and `c`")
    ## with conjunction and Oxford comma
    result = list_utils.cast_to_string(["a", "b", "c"], conjunction="and", use_oxford_comma=True)
    self.assertEqual(result, "`a`, `b`, and `c`")
    ## with only two elements
    result = list_utils.cast_to_string(["a", "b"])
    self.assertEqual(result, "`a` or `b`")
    result = list_utils.cast_to_string(["a", "b"], conjunction="")
    self.assertEqual(result, "`a`, `b`")
    ## single element
    result = list_utils.cast_to_string(["a"])
    self.assertEqual(result, "`a`")
    ## empty list
    result = list_utils.cast_to_string([])
    self.assertEqual(result, "")
    ## no quotes
    result = list_utils.cast_to_string(["x", "y"], wrap_in_quotes=False, conjunction="and")
    self.assertEqual(result, "x and y")

  def test_get_intersect_of_lists(self):
    ## intersection of two lists
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [2, 3, 4])
    list_expected = [2, 3]
    self.assertEqual(list_output, list_expected)
    ## no intersection
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [4, 5, 6])
    list_expected = []
    self.assertEqual(list_output, list_expected)
    ## identical lists
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## one list is empty
    list_output = list_utils.get_intersect_of_lists([], [1, 2, 3])
    list_expected = []
    self.assertEqual(list_output, list_expected)

  def test_get_union_of_lists(self):
    ## union of two lists
    list_output = list_utils.get_union_of_lists([1, 2, 3], [2, 3, 4])
    list_expected = [1, 2, 3, 4]
    self.assertEqual(list_output, list_expected)
    ## no common elements, just merge both lists
    list_output = list_utils.get_union_of_lists([1, 2, 3], [4, 5, 6])
    list_expected = [1, 2, 3, 4, 5, 6]
    self.assertEqual(list_output, list_expected)
    ## identical lists, should return one set of values
    list_output = list_utils.get_union_of_lists([1, 2, 3], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## one list is empty
    list_output = list_utils.get_union_of_lists([], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)

  def test_get_index_of_closest_value(self):
    ## typical case 1: target value exists in the list
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], 5)
    expected_index = 1
    self.assertEqual(output_index, expected_index)
    ## typical case 2: target value does not exist, find closest value
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], 6)
    expected_index = 1
    self.assertEqual(output_index, expected_index)
    ## target value is None
    with self.assertRaises(Exception):
      list_utils.get_index_of_closest_value([1, 2, 3], None)
    ## target value is infinity
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], numpy.inf)
    expected_index = 2
    self.assertEqual(output_index, expected_index)
    ## target value is negative infinity
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], -numpy.inf)
    expected_index = 0
    self.assertEqual(output_index, expected_index)

  def test_flatten_list(self):
    ## flatten a list of lists
    list_output = list_utils.flatten_list([[1, 2], [3, 4], [5, 6]])
    list_expected = [1, 2, 3, 4, 5, 6]
    self.assertEqual(list_output, list_expected)
    ## flatten an empty list of lists
    list_output = list_utils.flatten_list([[], []])
    list_expected = []
    self.assertEqual(list_output, list_expected)
    ## single list inside list, should return that list
    list_output = list_utils.flatten_list([[1, 2, 3]])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## already a flat list, should return the same list
    list_output = list_utils.flatten_list([1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)


## ###############################################################
## TEST ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST