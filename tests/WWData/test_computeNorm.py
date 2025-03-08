## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
import numpy
from Loki.WWData import SimpleStats


## ###############################################################
## TESTS
## ###############################################################
class Test_SimpleStats_computeNorm(unittest.TestCase):
  def test_l2_norm(self):
    # typical case: L2 norm (Euclidean distance)
    array_output = SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=2)
    expected_output = numpy.sqrt((4-1)**2 + (5-2)**2 + (6-3)**2)
    self.assertAlmostEqual(array_output, expected_output, places=5)
    # edge case 1: identical arrays, should return 0
    array_output = SimpleStats.computeNorm([1, 2, 3], [1, 2, 3], p=2)
    expected_output = 0
    self.assertEqual(array_output, expected_output)
    # edge case 2: empty arrays (raises error)
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([], [1, 2, 3], p=2)

  def test_l1_norm(self):
    # typical case: L1 norm (Manhattan distance)
    array_output = SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=1)
    expected_output = 9.0  # sum of absolute differences
    self.assertEqual(array_output, expected_output)
    # edge case 1: identical arrays, should return 0
    array_output = SimpleStats.computeNorm([1, 2, 3], [1, 2, 3], p=1)
    expected_output = 0
    self.assertEqual(array_output, expected_output)
    # edge case 2: empty arrays (raises error)
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([], [1, 2, 3], p=1)

  def test_l0_norm(self):
    # typical case: L0 norm (count non-zero differences)
    array_output = SimpleStats.computeNorm([1, 2, 3], [1, 3, 3], p=0)
    expected_output = 1  # only one element differs (2 vs 3)
    self.assertEqual(array_output, expected_output)
    # edge case 1: identical arrays, should return 0
    array_output = SimpleStats.computeNorm([1, 2, 3], [1, 2, 3], p=0)
    expected_output = 0
    self.assertEqual(array_output, expected_output)

  def test_infinity_norm(self):
    # typical case: infinity norm (maximum absolute difference)
    array_output = SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=numpy.inf)
    expected_output = 3  # max(3, 3, 3)
    self.assertEqual(array_output, expected_output)
    # edge case 1: identical arrays, should return 0
    array_output = SimpleStats.computeNorm([1, 2, 3], [1, 2, 3], p=numpy.inf)
    expected_output = 0
    self.assertEqual(array_output, expected_output)

  def test_normalisation(self):
    # typical case: normalise the L1 norm by length
    array_output = SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=1, bool_normalise=True)
    expected_output = ((4-1) + (5-2) + (6-3)) / 3
    self.assertAlmostEqual(array_output, expected_output, places=5)
    # typical case: normalise the L2 norm by length^1/2
    array_output = SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=2, bool_normalise=True)
    expected_output = numpy.sqrt((4-1)**2 + (5-2)**2 + (6-3)**2) / numpy.sqrt(3)
    self.assertAlmostEqual(array_output, expected_output, places=5)

  def test_invalid_p_value(self):
    # edge case 1: invalid p (negative value)
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p=-1)
    # edge case 2: invalid p (non-numeric value)
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([1, 2, 3], [4, 5, 6], p="invalid")

  def test_identical_arrays(self):
    # typical case: identical arrays, all norms should return 0
    for p in [0, 1, 2, numpy.inf]:
      array_output = SimpleStats.computeNorm([1, 2, 3], [1, 2, 3], p=p)
      expected_output = 0
      self.assertEqual(array_output, expected_output)

  def test_non_matching_shapes(self):
    # edge case: arrays of different shapes (raises error)
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([1, 2, 3], [1, 2], p=2)

  def test_empty_array(self):
    # edge case: one array is empty
    with self.assertRaises(ValueError):
      SimpleStats.computeNorm([], [1, 2, 3], p=2)

  def test_max_difference_zero(self):
    # edge case: max_diff = 0, the arrays are identical
    array_output = SimpleStats.computeNorm([0, 0, 0], [0, 0, 0], p=2)
    expected_output = 0
    self.assertEqual(array_output, expected_output)


## ###############################################################
## TEST ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST