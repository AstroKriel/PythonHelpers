## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest
from jormi.utils import list_utils

##
## === TEST SUITE
##


class TestListUtils(unittest.TestCase):

    def test_cast_to_string(self):
        result = list_utils.as_string(["a", "b", "c"])
        self.assertEqual(result, "`a`, `b`, or `c`")
        ## with conjunction but no Oxford comma
        result = list_utils.as_string(
            ["a", "b", "c"],
            conjunction="and",
            use_oxford_comma=False,
        )
        self.assertEqual(result, "`a`, `b` and `c`")
        ## with conjunction and Oxford comma
        result = list_utils.as_string(
            ["a", "b", "c"],
            conjunction="and",
            use_oxford_comma=True,
        )
        self.assertEqual(result, "`a`, `b`, and `c`")
        ## with only two elements
        result = list_utils.as_string(["a", "b"])
        self.assertEqual(result, "`a` or `b`")
        result = list_utils.as_string(["a", "b"], conjunction="")
        self.assertEqual(result, "`a`, `b`")
        ## single element
        result = list_utils.as_string(["a"])
        self.assertEqual(result, "`a`")
        ## empty list
        result = list_utils.as_string([])
        self.assertEqual(result, "")
        ## no quotes
        result = list_utils.as_string(["x", "y"], wrap_in_quotes=False, conjunction="and")
        self.assertEqual(result, "x and y")

    def test_get_intersect_of_lists(self):
        ## intersection of two lists
        output = list_utils.get_intersect_of_lists([1, 2, 3], [2, 3, 4])
        expected = [2, 3]
        self.assertEqual(output, expected)
        ## no intersection
        output = list_utils.get_intersect_of_lists([1, 2, 3], [4, 5, 6])
        expected = []
        self.assertEqual(output, expected)
        ## identical lists
        output = list_utils.get_intersect_of_lists([1, 2, 3], [1, 2, 3])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)
        ## one list is empty
        output = list_utils.get_intersect_of_lists([], [1, 2, 3])
        expected = []
        self.assertEqual(output, expected)

    def test_get_union_of_lists(self):
        ## union of two lists
        output = list_utils.get_union_of_lists([1, 2, 3], [2, 3, 4])
        expected = [1, 2, 3, 4]
        self.assertEqual(output, expected)
        ## no common elements, just merge both lists
        output = list_utils.get_union_of_lists([1, 2, 3], [4, 5, 6])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(output, expected)
        ## identical lists, should return one set of values
        output = list_utils.get_union_of_lists([1, 2, 3], [1, 2, 3])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)
        ## one list is empty
        output = list_utils.get_union_of_lists([], [1, 2, 3])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)

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
        output = list_utils.flatten_list([[1, 2], [3, 4, 5], [6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(output, expected)
        ## flatten a list of lists with mixed types
        output = list_utils.flatten_list([[1, "two"], [3, "four", 5], [[6, 7, 8]]])
        expected = [1, "two", 3, "four", 5, 6, 7, 8]
        self.assertEqual(output, expected)
        ## flatten an empty list of lists
        output = list_utils.flatten_list([[1, 2, [3]], []])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)
        ## single list inside list, should return that list
        output = list_utils.flatten_list([[1, 2, 3]])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)
        ## already a flat list, should return the same list
        output = list_utils.flatten_list([1, 2, 3])
        expected = [1, 2, 3]
        self.assertEqual(output, expected)
        ## list of numpy-arrays
        output = list_utils.flatten_list(
            [
                numpy.array([1, 2, 3]),
                numpy.array([4, 5, 6, 7]),
                numpy.array([8, 9]),
            ],
        )
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(output, expected)

    def test_get_index_of_first_crossing(self):
        ## rising crossing
        values = [0.1, 0.2, 0.5, 0.7, 1.0]
        index = list_utils.get_index_of_first_crossing(values, 0.6, direction="rising")
        self.assertEqual(index, 2)  # crosses between 0.5 (index 2) and 0.7 (index 3)
        ## falling crossing
        values = [1.0, 0.9, 0.6, 0.4, 0.2]
        index = list_utils.get_index_of_first_crossing(values, 0.5, direction="falling")
        self.assertEqual(index, 2)  # crosses between 0.6 and 0.4
        ## non-directional (any) crossing
        values = [0.9, 0.7, 0.4, 0.6, 0.8]
        index = list_utils.get_index_of_first_crossing(values, 0.5)
        self.assertEqual(index, 1)  # first crossing is falling: 0.7 - 0.4
        ## exact match value (rising)
        values = [0.1, 0.5, 0.6, 0.9]
        index = list_utils.get_index_of_first_crossing(values, 0.6, direction="rising")
        self.assertEqual(index, 1)  # 0.5 - 0.6
        ## target outside range (should raise)
        values = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError):
            list_utils.get_index_of_first_crossing(values, 1.0)
        ## invalid direction value (should raise)
        values = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError):
            list_utils.get_index_of_first_crossing(values, 0.2, direction="diagonal")
        ## exact min value match
        values = [0.1, 0.2, 0.3]
        index = list_utils.get_index_of_first_crossing(values, 0.1)
        self.assertEqual(index, 0)
        ## exact max value match
        values = [0.1, 0.5, 0.9]
        index = list_utils.get_index_of_first_crossing(values, 0.9)
        self.assertEqual(index, 2)
        ## no crossing found (returns none)
        values = [0.1, 0.2, 0.3]
        result = list_utils.get_index_of_first_crossing(values, 0.25, direction="falling")
        self.assertIsNone(result)

    def test_sample_returns_expected_length(self):

        def _test(_input_length, _sampled_length):
            elems = list(range(_input_length))
            out_elems = list_utils.sample_list(elems, _sampled_length)
            self.assertEqual(len(out_elems), _sampled_length)

        for input_length in range(18, 25):
            _test(_input_length=input_length, _sampled_length=5)

    def test_sample_expected_elems_divisible_cases(self):
        ## test: (num_elems - 1) % (max_elems - 1) == 0 -> last element included
        ## 20 % 4 == 0
        elems = list(range(21))
        out = list_utils.sample_list(elems, 5)
        expected = [0, 5, 10, 15, 20]
        self.assertEqual(out, expected)
        ## 12 % 4 == 0
        elems = list(range(13))
        out = list_utils.sample_list(elems, 5)
        expected = [0, 3, 6, 9, 12]
        self.assertEqual(out, expected)

    def test_sample_expected_elems_nondivisible_cases(self):
        ## last element is typically NOT included with pure integer stride
        ## 19 % 4 != 0
        elems = list(range(20))
        out = list_utils.sample_list(elems, 5)
        expected = [0, 4, 8, 12, 16]
        self.assertEqual(out, expected)
        self.assertNotEqual(out[-1], elems[-1])  # sanity check: last elem is not included
        ## 21 % 4 != 0
        elems = list(range(22))
        out = list_utils.sample_list(elems, 5)
        expected = [0, 5, 10, 15, 20]
        self.assertEqual(out, expected)
        self.assertNotEqual(out[-1], elems[-1])

    def test_first_element_and_constant_stringide(self):
        elems = list(range(37))
        max_elems = 7
        out = list_utils.sample_list(elems, max_elems)
        ## first element always
        self.assertEqual(out[0], elems[0])
        ## constant integer stride between chosen indices
        indices_to_keep = [value_index for value_index, value in enumerate(out)]
        gaps = [b - a for a, b in zip(indices_to_keep[:-1], indices_to_keep[1:])]
        self.assertTrue(all(g == gaps[0] for g in gaps))  # constant gap
        self.assertGreater(gaps[0], 0)

    def test_single_element_requested(self):
        elems = [1, 2, 3]
        out = list_utils.sample_list(elems, 1)
        self.assertEqual(out, [1])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            list_utils.sample_list([], 3)
        with self.assertRaises(ValueError):
            list_utils.sample_list([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            list_utils.sample_list([1, 2, 3], -1)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
