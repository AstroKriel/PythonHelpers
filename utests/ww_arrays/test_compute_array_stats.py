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

## simple normalised 1D PDF: 3 bins at [0, 1, 2] with uniform density 1/3
## edges at [-0.5, 0.5, 1.5, 2.5] -> widths all 1 -> integral = 1
_PDF_BIN_CENTERS = numpy.array([0.0, 1.0, 2.0])
_PDF_DENSITIES = numpy.full(3, 1.0 / 3.0)

## simple normalised 2D JPDF: 2x2 uniform density 0.25
## all bin widths 1 -> all bin areas 1 -> integral = 0.25 * 4 = 1
_JPDF_ROW_CENTERS = numpy.array([0.0, 1.0])
_JPDF_COL_CENTERS = numpy.array([0.0, 1.0])
_JPDF_DENSITIES = numpy.full((2, 2), 0.25)

##
## === TEST SUITES
##


class TestCheckNoZeroValues(unittest.TestCase):

    def test_clean_array_returns_false(
        self,
    ) -> None:
        result = compute_array_stats.check_no_zero_values(
            numpy.array([1.0, 2.0, 3.0]),
        )
        self.assertFalse(result)

    def test_array_with_zeros_raises(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.check_no_zero_values(
                numpy.array([1.0, 0.0, 3.0]),
            )

    def test_raise_error_false_returns_true(
        self,
    ) -> None:
        result = compute_array_stats.check_no_zero_values(
            numpy.array([1.0, 0.0, 3.0]),
            raise_error=False,
        )
        self.assertTrue(result)

    def test_raise_error_false_does_not_raise(
        self,
    ) -> None:
        compute_array_stats.check_no_zero_values(
            numpy.array([0.0]),
            raise_error=False,
        )


class TestCheckNoNonfiniteValues(unittest.TestCase):

    def test_clean_array_returns_false(
        self,
    ) -> None:
        result = compute_array_stats.check_no_nonfinite_values(
            numpy.array([1.0, 2.0, 3.0]),
        )
        self.assertFalse(result)

    def test_nan_raises(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.check_no_nonfinite_values(
                numpy.array([1.0, numpy.nan, 3.0]),
            )

    def test_posinf_raises(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.check_no_nonfinite_values(
                numpy.array([1.0, numpy.inf, 3.0]),
            )

    def test_neginf_raises(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.check_no_nonfinite_values(
                numpy.array([1.0, -numpy.inf, 3.0]),
            )

    def test_check_nan_false_ignores_nan(
        self,
    ) -> None:
        result = compute_array_stats.check_no_nonfinite_values(
            numpy.array([numpy.nan]),
            check_nan=False,
        )
        self.assertFalse(result)

    def test_check_posinf_false_ignores_posinf(
        self,
    ) -> None:
        result = compute_array_stats.check_no_nonfinite_values(
            numpy.array([numpy.inf]),
            check_posinf=False,
        )
        self.assertFalse(result)

    def test_check_neginf_false_ignores_neginf(
        self,
    ) -> None:
        result = compute_array_stats.check_no_nonfinite_values(
            numpy.array([-numpy.inf]),
            check_neginf=False,
        )
        self.assertFalse(result)

    def test_raise_error_false_returns_true(
        self,
    ) -> None:
        result = compute_array_stats.check_no_nonfinite_values(
            numpy.array([numpy.nan]),
            raise_error=False,
        )
        self.assertTrue(result)


class TestMakeNonfinitesZero(unittest.TestCase):

    def test_nan_replaced_with_zero(
        self,
    ) -> None:
        array = numpy.array([1.0, numpy.nan, 3.0])
        compute_array_stats.make_nonfinites_zero(array)
        self.assertEqual(
            array[1],
            0.0,
        )

    def test_posinf_replaced_with_zero(
        self,
    ) -> None:
        array = numpy.array([1.0, numpy.inf, 3.0])
        compute_array_stats.make_nonfinites_zero(array)
        self.assertEqual(
            array[1],
            0.0,
        )

    def test_neginf_replaced_with_zero(
        self,
    ) -> None:
        array = numpy.array([1.0, -numpy.inf, 3.0])
        compute_array_stats.make_nonfinites_zero(array)
        self.assertEqual(
            array[1],
            0.0,
        )

    def test_clean_array_unchanged(
        self,
    ) -> None:
        array = numpy.array([1.0, 2.0, 3.0])
        original = array.copy()
        compute_array_stats.make_nonfinites_zero(array)
        numpy.testing.assert_array_equal(
            array,
            original,
        )

    def test_zero_nan_false_leaves_nan(
        self,
    ) -> None:
        array = numpy.array([numpy.nan])
        compute_array_stats.make_nonfinites_zero(
            array,
            zero_nan=False,
        )
        self.assertTrue(
            numpy.isnan(array[0]),
        )

    def test_zero_posinf_false_leaves_posinf(
        self,
    ) -> None:
        array = numpy.array([numpy.inf])
        compute_array_stats.make_nonfinites_zero(
            array,
            zero_posinf=False,
        )
        self.assertTrue(
            numpy.isposinf(array[0]),
        )

    def test_zero_neginf_false_leaves_neginf(
        self,
    ) -> None:
        array = numpy.array([-numpy.inf])
        compute_array_stats.make_nonfinites_zero(
            array,
            zero_neginf=False,
        )
        self.assertTrue(
            numpy.isneginf(array[0]),
        )


class TestSuppressDivideWarnings(unittest.TestCase):

    def test_usable_as_context_manager(
        self,
    ) -> None:
        with compute_array_stats.suppress_divide_warnings():
            result = numpy.array([1.0]) / numpy.array([0.0])
        self.assertTrue(
            numpy.isposinf(result[0]),
        )

    def test_returns_context_manager(
        self,
    ) -> None:
        ctx = compute_array_stats.suppress_divide_warnings()
        self.assertTrue(
            hasattr(
                ctx,
                "__enter__",
            ),
        )
        self.assertTrue(
            hasattr(
                ctx,
                "__exit__",
            ),
        )


class TestComputeRms(unittest.TestCase):

    def test_uniform_array(
        self,
    ) -> None:
        self.assertAlmostEqual(
            compute_array_stats.compute_rms(
                numpy.full(
                    4,
                    3.0,
                ),
            ),
            3.0,
        )

    def test_zero_array(
        self,
    ) -> None:
        self.assertAlmostEqual(
            compute_array_stats.compute_rms(numpy.zeros(4)),
            0.0,
        )

    def test_known_case(
        self,
    ) -> None:
        ## rms([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5)
        self.assertAlmostEqual(
            compute_array_stats.compute_rms(numpy.array([3.0, 4.0])),
            numpy.sqrt(12.5),
        )

    def test_returns_float(
        self,
    ) -> None:
        self.assertIsInstance(
            compute_array_stats.compute_rms(numpy.array([1.0, 2.0])),
            float,
        )

    def test_sign_cancels_in_square(
        self,
    ) -> None:
        ## rms([-1, 1, -1, 1]) = 1.0
        self.assertAlmostEqual(
            compute_array_stats.compute_rms(numpy.array([-1.0, 1.0, -1.0, 1.0])),
            1.0,
        )


class TestEstimatedPDF_Construction(unittest.TestCase):

    def test_valid_construction(
        self,
    ) -> None:
        pdf = compute_array_stats.EstimatedPDF(
            bin_centers=_PDF_BIN_CENTERS,
            densities=_PDF_DENSITIES,
        )
        self.assertIsInstance(
            pdf,
            compute_array_stats.EstimatedPDF,
        )

    def test_rejects_non_1d_bin_centers(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedPDF(
                bin_centers=numpy.ones((2, 3)),
                densities=_PDF_DENSITIES,
            )

    def test_rejects_non_finite_bin_centers(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedPDF(
                bin_centers=numpy.array([0.0, numpy.nan, 2.0]),
                densities=_PDF_DENSITIES,
            )

    def test_rejects_mismatched_shapes(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedPDF(
                bin_centers=numpy.array([0.0, 1.0]),
                densities=_PDF_DENSITIES,
            )

    def test_rejects_unnormalised_densities(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.EstimatedPDF(
                bin_centers=_PDF_BIN_CENTERS,
                densities=numpy.array([1.0, 1.0, 1.0]),
            )


class TestEstimatedPDF_Properties(unittest.TestCase):

    def _make_pdf(
        self,
    ) -> compute_array_stats.EstimatedPDF:
        return compute_array_stats.EstimatedPDF(
            bin_centers=_PDF_BIN_CENTERS,
            densities=_PDF_DENSITIES,
        )

    def test_bin_edges_length(
        self,
    ) -> None:
        pdf = self._make_pdf()
        self.assertEqual(
            len(pdf.bin_edges),
            len(_PDF_BIN_CENTERS) + 1,
        )

    def test_bounds_lower_less_than_upper(
        self,
    ) -> None:
        pdf = self._make_pdf()
        lower, upper = pdf.bounds
        self.assertLess(
            lower,
            upper,
        )

    def test_bounds_are_floats(
        self,
    ) -> None:
        pdf = self._make_pdf()
        lower, upper = pdf.bounds
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)

    def test_resolution_equals_num_bins(
        self,
    ) -> None:
        pdf = self._make_pdf()
        self.assertEqual(
            pdf.resolution,
            len(_PDF_BIN_CENTERS),
        )


class TestEstimatedJPDF_Construction(unittest.TestCase):

    def test_valid_construction(
        self,
    ) -> None:
        jpdf = compute_array_stats.EstimatedJPDF(
            row_centers=_JPDF_ROW_CENTERS,
            col_centers=_JPDF_COL_CENTERS,
            densities=_JPDF_DENSITIES,
        )
        self.assertIsInstance(
            jpdf,
            compute_array_stats.EstimatedJPDF,
        )

    def test_rejects_non_1d_row_centers(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedJPDF(
                row_centers=numpy.ones((2, 2)),
                col_centers=_JPDF_COL_CENTERS,
                densities=_JPDF_DENSITIES,
            )

    def test_rejects_non_finite_col_centers(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedJPDF(
                row_centers=_JPDF_ROW_CENTERS,
                col_centers=numpy.array([0.0, numpy.nan]),
                densities=_JPDF_DENSITIES,
            )

    def test_rejects_non_2d_densities(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedJPDF(
                row_centers=_JPDF_ROW_CENTERS,
                col_centers=_JPDF_COL_CENTERS,
                densities=numpy.array([0.25, 0.25, 0.25, 0.25]),
            )

    def test_rejects_wrong_shape_densities(
        self,
    ) -> None:
        with self.assertRaises((TypeError, ValueError)):
            compute_array_stats.EstimatedJPDF(
                row_centers=_JPDF_ROW_CENTERS,
                col_centers=_JPDF_COL_CENTERS,
                densities=numpy.full((3, 3), 1.0 / 9.0),
            )

    def test_rejects_unnormalised_densities(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            compute_array_stats.EstimatedJPDF(
                row_centers=_JPDF_ROW_CENTERS,
                col_centers=_JPDF_COL_CENTERS,
                densities=numpy.ones((2, 2)),
            )


class TestEstimatedJPDF_Properties(unittest.TestCase):

    def _make_jpdf(
        self,
    ) -> compute_array_stats.EstimatedJPDF:
        return compute_array_stats.EstimatedJPDF(
            row_centers=_JPDF_ROW_CENTERS,
            col_centers=_JPDF_COL_CENTERS,
            densities=_JPDF_DENSITIES,
        )

    def test_row_edges_length(
        self,
    ) -> None:
        jpdf = self._make_jpdf()
        self.assertEqual(
            len(jpdf.row_edges),
            len(_JPDF_ROW_CENTERS) + 1,
        )

    def test_col_edges_length(
        self,
    ) -> None:
        jpdf = self._make_jpdf()
        self.assertEqual(
            len(jpdf.col_edges),
            len(_JPDF_COL_CENTERS) + 1,
        )

    def test_bounds_col_lower_less_than_upper(
        self,
    ) -> None:
        jpdf = self._make_jpdf()
        (col_lower, col_upper), _ = jpdf.bounds
        self.assertLess(
            col_lower,
            col_upper,
        )

    def test_bounds_row_lower_less_than_upper(
        self,
    ) -> None:
        jpdf = self._make_jpdf()
        _, (row_lower, row_upper) = jpdf.bounds
        self.assertLess(
            row_lower,
            row_upper,
        )

    def test_resolution_matches_centers(
        self,
    ) -> None:
        jpdf = self._make_jpdf()
        self.assertEqual(
            jpdf.resolution,
            (len(_JPDF_ROW_CENTERS), len(_JPDF_COL_CENTERS)),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
