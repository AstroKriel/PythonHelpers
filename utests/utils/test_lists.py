## { TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi import ww_lists

##
## === TEST SUITE
##


class Tests(unittest.TestCase):

    def test_cast_to_string(
        self,
    ):
        ## default: no quotes, no conjunction
        result = ww_lists.as_string(["a", "b", "c"])
        self.assertEqual(
            result,
            "a, b, c",
        )
        ## with conjunction
        result = ww_lists.as_string(["a", "b", "c"], conjunction="and")
        self.assertEqual(
            result,
            "a, b and c",
        )
        ## with conjunction and quotes
        result = ww_lists.as_string(["a", "b", "c"], wrap_in_quotes=True, conjunction="or")
        self.assertEqual(
            result,
            "`a`, `b` or `c`",
        )
        ## two elements with conjunction
        result = ww_lists.as_string(["a", "b"], conjunction="or")
        self.assertEqual(
            result,
            "a or b",
        )
        ## two elements no conjunction
        result = ww_lists.as_string(["a", "b"])
        self.assertEqual(
            result,
            "a, b",
        )
        ## single element
        result = ww_lists.as_string(["a"])
        self.assertEqual(
            result,
            "a",
        )
        ## empty list
        result = ww_lists.as_string([])
        self.assertEqual(
            result,
            "",
        )
        ## no quotes, conjunction
        result = ww_lists.as_string(["x", "y"], wrap_in_quotes=False, conjunction="and")
        self.assertEqual(
            result,
            "x and y",
        )

    def test_get_intersect_of_lists(
        self,
    ):
        ## intersection of two lists
        output = ww_lists.get_intersect_of_lists(
            list_a=[1, 2, 3],
            list_b=[2, 3, 4],
        )
        expected = [2, 3]
        self.assertEqual(
            sorted(output),
            sorted(expected),
        )
        ## no intersection
        output = ww_lists.get_intersect_of_lists(
            list_a=[1, 2, 3],
            list_b=[4, 5, 6],
        )
        self.assertEqual(
            output,
            [],
        )
        ## identical lists
        output = ww_lists.get_intersect_of_lists(
            list_a=[1, 2, 3],
            list_b=[1, 2, 3],
        )
        self.assertEqual(
            sorted(output),
            [1, 2, 3],
        )
        ## one list is empty
        output = ww_lists.get_intersect_of_lists(
            list_a=[],
            list_b=[1, 2, 3],
        )
        self.assertEqual(
            output,
            [],
        )

    def test_get_union_of_lists(
        self,
    ):
        ## union of two lists
        output = ww_lists.get_union_of_lists(
            list_a=[1, 2, 3],
            list_b=[2, 3, 4],
        )
        self.assertEqual(
            sorted(output),
            [1, 2, 3, 4],
        )
        ## no common elements, just merge both lists
        output = ww_lists.get_union_of_lists(
            list_a=[1, 2, 3],
            list_b=[4, 5, 6],
        )
        self.assertEqual(
            sorted(output),
            [1, 2, 3, 4, 5, 6],
        )
        ## identical lists, should return one set of values
        output = ww_lists.get_union_of_lists(
            list_a=[1, 2, 3],
            list_b=[1, 2, 3],
        )
        self.assertEqual(
            sorted(output),
            [1, 2, 3],
        )
        ## one list is empty
        output = ww_lists.get_union_of_lists(
            list_a=[],
            list_b=[1, 2, 3],
        )
        self.assertEqual(
            output,
            [1, 2, 3],
        )

    def test_get_index_of_closest_value(
        self,
    ):
        ## typical case 1: target value exists in the list
        output_index = ww_lists.get_index_of_closest_value(
            values=[1, 5, 8],
            target=5,
        )
        self.assertEqual(
            output_index,
            1,
        )
        ## typical case 2: target value does not exist, find closest value
        output_index = ww_lists.get_index_of_closest_value(
            values=[1, 5, 8],
            target=6,
        )
        self.assertEqual(
            output_index,
            1,
        )
        ## target value is None
        with self.assertRaises(Exception):
            ww_lists.get_index_of_closest_value(
                values=[1, 2, 3],
                target=None,  # type: ignore
            )
        ## target value is infinity
        output_index = ww_lists.get_index_of_closest_value(
            values=[1, 5, 8],
            target=numpy.inf,
        )
        self.assertEqual(
            output_index,
            2,
        )
        ## target value is negative infinity
        output_index = ww_lists.get_index_of_closest_value(
            values=[1, 5, 8],
            target=-numpy.inf,
        )
        self.assertEqual(
            output_index,
            0,
        )

    def test_flatten_list(
        self,
    ):
        ## flatten a list of lists
        output = ww_lists.flatten_list([[1, 2], [3, 4, 5], [6]])
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(
            output,
            expected,
        )
        ## flatten a list of lists with mixed types
        output = ww_lists.flatten_list([[1, "two"], [3, "four", 5], [[6, 7, 8]]])
        expected = [1, "two", 3, "four", 5, 6, 7, 8]
        self.assertEqual(
            output,
            expected,
        )
        ## flatten an empty list of lists
        output = ww_lists.flatten_list([[1, 2, [3]], []])
        expected = [1, 2, 3]
        self.assertEqual(
            output,
            expected,
        )
        ## single list inside list, should return that list
        output = ww_lists.flatten_list([[1, 2, 3]])
        expected = [1, 2, 3]
        self.assertEqual(
            output,
            expected,
        )
        ## already a flat list, should return the same list
        output = ww_lists.flatten_list([1, 2, 3])
        expected = [1, 2, 3]
        self.assertEqual(
            output,
            expected,
        )
        ## list of numpy-arrays: numpy arrays are not lists so they are not expanded
        output = ww_lists.flatten_list([
            numpy.array([1, 2, 3]),
            numpy.array([4, 5, 6, 7]),
        ], )
        self.assertEqual(
            len(output),
            2,
        )

    def test_get_index_of_first_crossing(
        self,
    ):
        ## rising crossing
        values = [0.1, 0.2, 0.5, 0.7, 1.0]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.6,
            direction="rising",
        )
        self.assertEqual(index, 2)  # crosses between 0.5 (index 2) and 0.7 (index 3)
        ## falling crossing
        values = [1.0, 0.9, 0.6, 0.4, 0.2]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.5,
            direction="falling",
        )
        self.assertEqual(index, 2)  # crosses between 0.6 and 0.4
        ## non-directional (any) crossing
        values = [0.9, 0.7, 0.4, 0.6, 0.8]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.5,
        )
        self.assertEqual(index, 1)  # first crossing is falling: 0.7 - 0.4
        ## exact match value (rising)
        values = [0.1, 0.5, 0.6, 0.9]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.6,
            direction="rising",
        )
        self.assertEqual(index, 1)  # 0.5 - 0.6
        ## target outside range (should raise)
        values = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError):
            ww_lists.get_index_of_first_crossing(
                values=values,
                target=1.0,
            )
        ## invalid direction value (should raise)
        values = [0.1, 0.2, 0.3]
        with self.assertRaises(ValueError):
            ww_lists.get_index_of_first_crossing(
                values=values,
                target=0.2,
                direction="diagonal",  # type: ignore
            )  # type: ignore
        ## exact min value match
        values = [0.1, 0.2, 0.3]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.1,
        )
        self.assertEqual(index, 0)
        ## exact max value match
        values = [0.1, 0.5, 0.9]
        index = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.9,
        )
        self.assertEqual(index, 2)
        ## no crossing found (returns none)
        values = [0.1, 0.2, 0.3]
        result = ww_lists.get_index_of_first_crossing(
            values=values,
            target=0.25,
            direction="falling",
        )
        self.assertIsNone(result)

    def test_sample_returns_expected_length(
        self,
    ):

        def _test(
            _input_length: int,
            _sampled_length: int,
        ) -> None:
            elems = list(range(_input_length))
            out_elems = ww_lists.sample_list(
                elems=elems,
                max_elems=_sampled_length,
            )
            self.assertEqual(
                len(out_elems),
                _sampled_length,
            )

        for input_length in range(18, 25):
            _test(
                _input_length=input_length,
                _sampled_length=5,
            )

    def test_sample_expected_elems_divisible_cases(
        self,
    ):
        ## test: (num_elems - 1) % (max_elems - 1) == 0 -> last element included
        ## 20 % 4 == 0
        elems = list(range(21))
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=5,
        )
        self.assertEqual(out, [0, 5, 10, 15, 20])
        ## 12 % 4 == 0
        elems = list(range(13))
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=5,
        )
        self.assertEqual(
            out,
            [0, 3, 6, 9, 12],
        )

    def test_sample_expected_elems_nondivisible_cases(
        self,
    ):
        ## last element is typically NOT included with pure integer stride
        ## 19 % 4 != 0
        elems = list(range(20))
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=5,
        )
        self.assertEqual(
            out,
            [0, 4, 8, 12, 16],
        )
        self.assertNotEqual(
            out[-1],
            elems[-1],
        )  # sanity check: last elem is not included
        ## 21 % 4 != 0
        elems = list(range(22))
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=5,
        )
        self.assertEqual(
            out,
            [0, 5, 10, 15, 20],
        )
        self.assertNotEqual(
            out[-1],
            elems[-1],
        )

    def test_first_element_and_constant_stringide(
        self,
    ):
        elems = list(range(37))
        max_elems = 7
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=max_elems,
        )
        ## first element always
        self.assertEqual(out[0], elems[0])
        ## constant integer stride between chosen indices
        indices_to_keep = [value_index for value_index, _value in enumerate(out)]
        gaps = [b - a for a, b in zip(indices_to_keep[:-1], indices_to_keep[1:])]
        self.assertTrue(all(g == gaps[0] for g in gaps))  # constant gap
        self.assertGreater(
            gaps[0],
            0,
        )

    def test_single_element_requested(
        self,
    ):
        elems = [1, 2, 3]
        out = ww_lists.sample_list(
            elems=elems,
            max_elems=1,
        )
        self.assertEqual(out, [1])

    def test_invalid_inputs(
        self,
    ):
        with self.assertRaises(ValueError):
            ww_lists.sample_list(
                elems=[],
                max_elems=3,
            )
        with self.assertRaises(ValueError):
            ww_lists.sample_list(
                elems=[1, 2, 3],
                max_elems=-1,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
