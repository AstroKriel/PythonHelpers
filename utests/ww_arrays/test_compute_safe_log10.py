## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays import compute_array_stats

##
## === TEST SUITE
##


class TestComputeSafeLog10(unittest.TestCase):

    def test_positive_values(
        self,
    ) -> None:
        result = compute_array_stats.compute_safe_log10(
            numpy.array([1.0, 10.0, 100.0], ),
        )
        numpy.testing.assert_array_almost_equal(
            result,
            [0.0, 1.0, 2.0],
        )

    def test_zero_gives_nan(
        self,
    ) -> None:
        result = compute_array_stats.compute_safe_log10(
            numpy.array([0.0], ),
        )
        self.assertTrue(
            numpy.isnan(
                result[0],
            ),
        )

    def test_negative_gives_nan(
        self,
    ) -> None:
        result = compute_array_stats.compute_safe_log10(
            numpy.array([-1.0, -100.0], ),
        )
        self.assertTrue(
            numpy.all(
                numpy.isnan(
                    result,
                ),
            ),
        )

    def test_mixed_array(
        self,
    ) -> None:
        result = compute_array_stats.compute_safe_log10(
            numpy.array([-1.0, 0.0, 10.0], ),
        )
        self.assertTrue(
            numpy.isnan(
                result[0],
            ),
        )
        self.assertTrue(
            numpy.isnan(
                result[1],
            ),
        )
        self.assertAlmostEqual(
            result[2],
            1.0,
        )

    def test_integer_array_returns_float(
        self,
    ) -> None:
        result = compute_array_stats.compute_safe_log10(
            numpy.array([1, 10, 100], ),
        )
        self.assertEqual(
            result.dtype,
            numpy.float64,
        )

    def test_output_shape_matches_input(
        self,
    ) -> None:
        array = numpy.ones((3, 4))
        result = compute_array_stats.compute_safe_log10(array)
        self.assertEqual(
            result.shape,
            array.shape,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
