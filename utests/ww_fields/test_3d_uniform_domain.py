## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_fields.fields_3d import domain_types

##
## === TEST SUITES
##


class TestConstruction(unittest.TestCase):

    def test_constructs_valid_3d_domain(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, False),
            resolution=(8, 4, 2),
            domain_bounds=((0.0, 1.0), (-2.0, 2.0), (10.0, 12.0)),
        )
        self.assertEqual(
            udomain_3d.num_sdims,
            3,
        )
        self.assertEqual(
            udomain_3d.periodicity,
            (True, True, False),
        )
        self.assertEqual(
            udomain_3d.resolution,
            (8, 4, 2),
        )
        self.assertEqual(
            udomain_3d.domain_bounds,
            ((0.0, 1.0), (-2.0, 2.0), (10.0, 12.0)),
        )

    def test_rejects_num_sdims_argument(
        self,
    ):
        with self.assertRaises(TypeError):
            domain_types.UniformDomain_3D(
                num_sdims=3, # type: ignore
                periodicity=(True, False, True),
                resolution=(8, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            )

    def test_resolution_length_must_be_3(
        self,
    ):
        with self.assertRaises((TypeError, ValueError)):
            domain_types.UniformDomain_3D(
                periodicity=(True, True, True),
                resolution=(8, 4), # type: ignore
                domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            )

    def test_domain_bounds_length_must_be_3(
        self,
    ):
        with self.assertRaises((TypeError, ValueError)):
            domain_types.UniformDomain_3D(
                periodicity=(True, True, True),
                resolution=(8, 4, 2),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),  # type: ignore
            )


class TestProperties(unittest.TestCase):

    def test_lengths_widths_volume_total_volume(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(10, 4, 2),
            domain_bounds=((0.0, 2.0), (-1.0, 3.0), (0.0, 1.0)),
        )
        self.assertEqual(
            udomain_3d.domain_lengths,
            (2.0, 4.0, 1.0),
        )
        self.assertEqual(
            udomain_3d.cell_widths,
            (0.2, 1.0, 0.5),
        )
        self.assertAlmostEqual(
            udomain_3d.cell_volume,
            0.2 * 1.0 * 0.5,
        )
        self.assertAlmostEqual(
            udomain_3d.total_volume,
            2.0 * 4.0 * 1.0,
        )
        self.assertEqual(
            udomain_3d.num_cells,
            10 * 4 * 2,
        )

    def test_cell_centers_shapes_and_values(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(4, 2, 2),
            domain_bounds=((0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)),
        )
        x0_centers, x1_centers, x2_centers = udomain_3d.cell_centers
        self.assertEqual(
            x0_centers.shape,
            (4, ),
        )
        self.assertEqual(
            x1_centers.shape,
            (2, ),
        )
        self.assertEqual(
            x2_centers.shape,
            (2, ),
        )
        expected_x0 = numpy.array([0.125, 0.375, 0.625, 0.875])
        expected_x1 = numpy.array([0.5, 1.5])
        expected_x2 = numpy.array([-0.5, 0.5])
        self.assertTrue(
            numpy.allclose(
                x0_centers,
                expected_x0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                x1_centers,
                expected_x1,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                x2_centers,
                expected_x2,
            ),
        )

    def test_cached_properties_return_same_object(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(3, 3, 3),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        self.assertIs(
            udomain_3d.cell_widths,
            udomain_3d.cell_widths,
        )
        self.assertIs(
            udomain_3d.domain_lengths,
            udomain_3d.domain_lengths,
        )
        self.assertIs(
            udomain_3d.cell_centers,
            udomain_3d.cell_centers,
        )


class TestEnsureHelpers(unittest.TestCase):

    def test_ensure_3d_udomain_accepts(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        domain_types.ensure_3d_udomain(udomain_3d=udomain_3d)

    def test_ensure_3d_udomain_rejects_wrong_type(
        self,
    ):
        with self.assertRaises(TypeError):
            domain_types.ensure_3d_udomain(udomain_3d=None)  # type: ignore

    def test_ensure_3d_periodic_udomain_accepts_fully_periodic(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        domain_types.ensure_3d_periodic_udomain(udomain_3d=udomain_3d)

    def test_ensure_3d_periodic_udomain_rejects_not_fully_periodic(
        self,
    ):
        udomain_3d = domain_types.UniformDomain_3D(
            periodicity=(True, False, True),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        with self.assertRaises(ValueError):
            domain_types.ensure_3d_periodic_udomain(udomain_3d=udomain_3d)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
