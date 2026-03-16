## { TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_fields import (
    _domain_types,
    cartesian_axes,
)

##
## === TEST SUITES
##


class TestConstruction(unittest.TestCase):

    def test_constructs_valid_1d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=1,
            periodicity=(True, ),
            resolution=(4, ),
            domain_bounds=((0.0, 1.0), ),
        )
        self.assertEqual(
            udomain.num_sdims,
            1,
        )
        self.assertEqual(
            udomain.periodicity,
            (True, ),
        )
        self.assertEqual(
            udomain.resolution,
            (4, ),
        )
        self.assertEqual(
            udomain.domain_bounds,
            ((0.0, 1.0), ),
        )

    def test_constructs_valid_2d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=2,
            periodicity=(True, False),
            resolution=(8, 4),
            domain_bounds=((0.0, 1.0), (-2.0, 2.0)),
        )
        self.assertEqual(
            udomain.num_sdims,
            2,
        )
        self.assertEqual(
            udomain.num_cells,
            8 * 4,
        )

    def test_constructs_valid_3d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=3,
            periodicity=(True, True, True),
            resolution=(4, 5, 6),
            domain_bounds=((0.0, 1.0), (0.0, 2.0), (-3.0, 3.0)),
        )
        self.assertEqual(
            udomain.num_sdims,
            3,
        )
        self.assertEqual(
            udomain.num_cells,
            4 * 5 * 6,
        )


class TestValidation(unittest.TestCase):

    def test_rejects_invalid_num_sdims(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=0,
                periodicity=(),
                resolution=(),
                domain_bounds=(),
            )
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=-1,
                periodicity=(),
                resolution=(),
                domain_bounds=(),
            )
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=1.0, # type: ignore[arg-type]
                periodicity=(True,),
                resolution=(4,),
                domain_bounds=((0.0, 1.0),),
            )

    def test_rejects_wrong_periodicity_length(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, True, True),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            )

    def test_rejects_non_bool_periodicity(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, 1), # type: ignore[arg-type]
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            )

    def test_rejects_wrong_resolution_length(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4, 4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            )

    def test_rejects_non_int_resolution(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4.0, 4), # type: ignore[arg-type]
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            )

    def test_rejects_non_positive_resolution_and_mentions_axis_label(self):
        with self.assertRaises(ValueError) as assert_context:
            _domain_types.UniformDomain(
                num_sdims=3,
                periodicity=(True, True, True),
                resolution=(4, 0, 6),
                domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            )
        assert_message = str(assert_context.exception)
        self.assertIn(
            "<resolution>",
            assert_message,
        )
        self.assertIn(
            cartesian_axes.get_axis_label(axis=1),
            assert_message,
        )

    def test_rejects_wrong_domain_bounds_length(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), ),
            )

    def test_rejects_domain_bounds_not_pairs(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0, 2.0), (0.0, 1.0)),  # type: ignore[arg-type]
            )

    def test_rejects_domain_bounds_non_numeric(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4, 4),
                domain_bounds=(("a", "b"), (0.0, 1.0)),  # type: ignore[arg-type]
            )

    def test_rejects_domain_bounds_hi_not_greater_than_lo_and_mentions_axis_label(self):
        with self.assertRaises(ValueError) as assert_context:
            _domain_types.UniformDomain(
                num_sdims=2,
                periodicity=(True, False),
                resolution=(4, 4),
                domain_bounds=((0.0, 0.0), (0.0, 1.0)),
            )
        assert_message = str(assert_context.exception)
        self.assertIn(cartesian_axes.get_axis_label(axis=0), assert_message)
        self.assertIn("max bound must be > min bound", assert_message)

    def test_rejects_num_sdims_exceeding_supported_cartesian_axes(self):
        with self.assertRaises((TypeError, ValueError)):
            _domain_types.UniformDomain(
                num_sdims=4,
                periodicity=(True, True, True, True),
                resolution=(4, 4, 4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            )


class TestProperties(unittest.TestCase):

    def test_lengths_widths_measures_num_cells_2d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=2,
            periodicity=(True, False),
            resolution=(10, 4),
            domain_bounds=((0.0, 2.0), (-1.0, 3.0)),
        )
        self.assertEqual(
            udomain.domain_lengths,
            (2.0, 4.0),
        )
        self.assertEqual(
            udomain.cell_widths,
            (0.2, 1.0),
        )
        self.assertAlmostEqual(
            udomain._measure_per_cell,
            0.2 * 1.0,
        )
        self.assertAlmostEqual(
            udomain._total_measure,
            2.0 * 4.0,
        )
        self.assertEqual(
            udomain.num_cells,
            40,
        )

    def test_lengths_widths_measures_num_cells_3d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=3,
            periodicity=(True, True, True),
            resolution=(2, 3, 4),
            domain_bounds=((0.0, 1.0), (0.0, 3.0), (-2.0, 2.0)),
        )
        self.assertEqual(
            udomain.domain_lengths,
            (1.0, 3.0, 4.0),
        )
        self.assertEqual(
            udomain.cell_widths,
            (0.5, 1.0, 1.0),
        )
        self.assertAlmostEqual(
            udomain._measure_per_cell,
            0.5 * 1.0 * 1.0,
        )
        self.assertAlmostEqual(
            udomain._total_measure,
            1.0 * 3.0 * 4.0,
        )
        self.assertEqual(
            udomain.num_cells,
            2 * 3 * 4,
        )

    def test_cell_centers_values_1d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=1,
            periodicity=(True, ),
            resolution=(4, ),
            domain_bounds=((0.0, 1.0), ),
        )
        cell_centers = udomain.cell_centers
        self.assertEqual(len(cell_centers), 1)
        expected = numpy.array([0.125, 0.375, 0.625, 0.875], dtype=float)
        self.assertTrue(
            numpy.allclose(
                cell_centers[0],
                expected,
            ),
        )

    def test_cell_centers_shapes_3d(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=3,
            periodicity=(True, True, True),
            resolution=(3, 5, 7),
            domain_bounds=((0.0, 3.0), (10.0, 20.0), (-7.0, 0.0)),
        )
        x0_centers, x1_centers, x2_centers = udomain.cell_centers
        self.assertEqual(
            x0_centers.shape,
            (3, ),
        )
        self.assertEqual(
            x1_centers.shape,
            (5, ),
        )
        self.assertEqual(
            x2_centers.shape,
            (7, ),
        )
        self.assertTrue(
            numpy.all((x1_centers > 10.0) & (x1_centers < 20.0)),
        )
        self.assertTrue(
            numpy.all((x2_centers > -7.0) & (x2_centers < 0.0)),
        )

    def test_cached_properties_return_same_object(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=2,
            periodicity=(True, True),
            resolution=(3, 3),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        )
        self.assertIs(
            udomain.cell_widths,
            udomain.cell_widths,
        )
        self.assertIs(
            udomain.domain_lengths,
            udomain.domain_lengths,
        )
        self.assertIs(
            udomain.cell_centers,
            udomain.cell_centers,
        )


class TestEnsureHelpers(unittest.TestCase):

    def test_ensure_udomain_accepts(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=2,
            periodicity=(True, False),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        )
        _domain_types.ensure_udomain(udomain=udomain)

    def test_ensure_udomain_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            _domain_types.ensure_udomain(udomain=None)  # type: ignore[arg-type]

    def test_ensure_udomain_metadata_num_sdims(self):
        udomain = _domain_types.UniformDomain(
            num_sdims=2,
            periodicity=(True, False),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        )
        _domain_types.ensure_udomain_metadata(
            udomain=udomain,
            num_sdims=2,
        )
        with self.assertRaises(ValueError):
            _domain_types.ensure_udomain_metadata(
                udomain=udomain,
                num_sdims=3,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
