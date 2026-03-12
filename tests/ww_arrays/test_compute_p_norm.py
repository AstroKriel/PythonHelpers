## { TEST

##
## === DEPENDENCIES
##

import unittest
import numpy
from jormi.ww_data import compute_stats

##
## === TEST SUITE
##


class Test_Compute_P_Norm(unittest.TestCase):

    def test_l2_norm(self):
        ## Euclidean distance
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[4, 5, 6],
            p_norm=2,
        )
        expected_output = numpy.sqrt((4 - 1)**2 + (5 - 2)**2 + (6 - 3)**2)
        self.assertAlmostEqual(array_output, expected_output, places=5)
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[1, 2, 3],
            p_norm=2,
        )
        expected_output = 0
        self.assertEqual(array_output, expected_output)
        with self.assertRaises(ValueError):
            compute_stats.compute_p_norm(
                array_a=[],
                array_b=[1, 2, 3],
                p_norm=2,
            )

    def test_l1_norm(self):
        ## Manhattan distance
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[4, 5, 6],
            p_norm=1,
        )
        expected_output = 9.0  # sum of absolute differences
        self.assertEqual(array_output, expected_output)
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[1, 2, 3],
            p_norm=1,
        )
        expected_output = 0
        self.assertEqual(array_output, expected_output)

    def test_l0_norm(self):
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 0, 3],
            array_b=[1, 3, 3],
            p_norm=0,
        )
        expected_output = 1  # only one element differs (2 vs 3)
        self.assertEqual(array_output, expected_output)
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[1, 2, 3],
            p_norm=0,
        )
        expected_output = 0
        self.assertEqual(array_output, expected_output)

    def test_infinity_norm(self):
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[4, 5, 6],
            p_norm=numpy.inf,
        )
        expected_output = 3  # max(3, 3, 3)
        self.assertEqual(array_output, expected_output)
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[1, 2, 3],
            p_norm=numpy.inf,
        )
        expected_output = 0
        self.assertEqual(array_output, expected_output)

    def test_normalisation(self):
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[4, 5, 6],
            p_norm=1,
            normalise_by_length=True,
        )
        expected_output = ((4 - 1) + (5 - 2) + (6 - 3)) / 3
        self.assertAlmostEqual(array_output, expected_output, places=5)
        array_output = compute_stats.compute_p_norm(
            array_a=[1, 2, 3],
            array_b=[4, 5, 6],
            p_norm=2,
            normalise_by_length=True,
        )
        expected_output = numpy.sqrt((4 - 1)**2 + (5 - 2)**2 + (6 - 3)**2) / numpy.sqrt(3)
        self.assertAlmostEqual(array_output, expected_output, places=5)

    def test_invalid_norm_order(self):
        with self.assertRaises(ValueError):
            compute_stats.compute_p_norm(
                array_a=[1, 2, 3],
                array_b=[4, 5, 6],
                p_norm=-1,
            )
        with self.assertRaises(ValueError):
            compute_stats.compute_p_norm(
                array_a=[1, 2, 3],
                array_b=[4, 5, 6],
                p_norm="invalid",  # type: ignore
            )

    def test_identical_arrays(self):
        for norm_order in [0, 1, 2, numpy.inf]:
            array_output = compute_stats.compute_p_norm(
                array_a=[1, 2, 3],
                array_b=[1, 2, 3],
                p_norm=norm_order,
            )
            expected_output = 0
            self.assertEqual(array_output, expected_output)

    def test_empty_arrays(self):
        for norm_order in [0, 1, 2, numpy.inf]:
            with self.assertRaises(ValueError):
                compute_stats.compute_p_norm(
                    array_a=[],
                    array_b=[1, 2, 3],
                    p_norm=norm_order,
                )

    def test_non_matching_array_shapes(self):
        with self.assertRaises(ValueError):
            compute_stats.compute_p_norm(
                array_a=[1, 2, 3],
                array_b=[1, 2],
                p_norm=2,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
