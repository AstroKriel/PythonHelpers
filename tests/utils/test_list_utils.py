## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
import numpy
from loki.utils import list_utils


## ###############################################################
## TESTS
## ###############################################################
class TestListUtils(unittest.TestCase):

  def test_getIntersectOfLists(self):
    ## typical case: intersection of two lists
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [2, 3, 4])
    list_expected = [2, 3]
    self.assertEqual(list_output, list_expected)
    ## edge case 1: no intersection
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [4, 5, 6])
    list_expected = []
    self.assertEqual(list_output, list_expected)
    ## edge case 2: identical lists
    list_output = list_utils.get_intersect_of_lists([1, 2, 3], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## edge case 3: one list is empty
    list_output = list_utils.get_intersect_of_lists([], [1, 2, 3])
    list_expected = []
    self.assertEqual(list_output, list_expected)

  def test_getUnionOfLists(self):
    ## typical case: union of two lists
    list_output = list_utils.get_union_of_lists([1, 2, 3], [2, 3, 4])
    list_expected = [1, 2, 3, 4]
    self.assertEqual(list_output, list_expected)
    ## edge case 1: no common elements, just merge both lists
    list_output = list_utils.get_union_of_lists([1, 2, 3], [4, 5, 6])
    list_expected = [1, 2, 3, 4, 5, 6]
    self.assertEqual(list_output, list_expected)
    ## edge case 2: identical lists, should return one set of values
    list_output = list_utils.get_union_of_lists([1, 2, 3], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## edge case 3: one list is empty
    list_output = list_utils.get_union_of_lists([], [1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)

  def test_getIndexOfClosestValue(self):
    ## typical case 1: target value exists in the list
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], 5)
    expected_index = 1
    self.assertEqual(output_index, expected_index)
    ## typical case 2: target value does not exist, find closest value
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], 6)
    expected_index = 1
    self.assertEqual(output_index, expected_index)
    ## edge case 1: target value is None
    with self.assertRaises(Exception):
      list_utils.get_index_of_closest_value([1, 2, 3], None)
    ## edge case 2: target value is infinity
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], numpy.inf)
    expected_index = 2
    self.assertEqual(output_index, expected_index)
    ## edge case 3: target value is negative infinity
    output_index = list_utils.get_index_of_closest_value([1, 5, 8], -numpy.inf)
    expected_index = 0
    self.assertEqual(output_index, expected_index)

  def test_flattenList(self):
    ## typical case: flatten a list of lists
    list_output = list_utils.flatten_list([[1, 2], [3, 4], [5, 6]])
    list_expected = [1, 2, 3, 4, 5, 6]
    self.assertEqual(list_output, list_expected)
    ## edge case 1: flatten an empty list of lists
    list_output = list_utils.flatten_list([[], []])
    list_expected = []
    self.assertEqual(list_output, list_expected)
    ## edge case 2: single list inside list, should return that list
    list_output = list_utils.flatten_list([[1, 2, 3]])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)
    ## edge case 3: already a flat list, should return the same list
    list_output = list_utils.flatten_list([1, 2, 3])
    list_expected = [1, 2, 3]
    self.assertEqual(list_output, list_expected)


## ###############################################################
## TEST ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST