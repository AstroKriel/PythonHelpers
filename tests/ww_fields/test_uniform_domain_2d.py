## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest

from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_2d import domain_type

##
## === TEST SUITES
##


class TestConstruction(unittest.TestCase):

    def test_constructs_valid_2d_domain(self):
        udomain_2d = domain_type.UniformDomain_2D(
            periodicity=(True, False),
            resolution=(8, 4),
            domain_bounds=((0.0, 1.0), (-2.0, 2.0)),
        )
        self.assertEqual(
            udomain_2d.num_sdims,
            2,
        )
        self.assertEqual(
            udomain_2d.periodicity,
            (True, False),
        )
        self.assertEqual(
            udomain_2d.resolution,
            (8, 4),
        )
        self.assertEqual(
            udomain_2d.domain_bounds,
            ((0.0, 1.0), (-2.0, 2.0)),
        )

    def test_rejects_num_sdims_argument(self):
        with self.assertRaises(TypeError):
            domain_type.UniformDomain_2D(
                num_sdims=2, # type: ignore[call-arg]
                periodicity=(True, False),
                resolution=(8, 4),
                domain_bounds=((0.0, 1.0), (-2.0, 2.0)),
            )

    def test_resolution_length_must_be_2(self):
        with self.assertRaises((TypeError, ValueError)):
            domain_type.UniformDomain_2D(
                periodicity=(True, False),
                resolution=(8, 4, 2),  # type: ignore[arg-type]
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            )

    def test_domain_bounds_length_must_be_2(self):
        with self.assertRaises((TypeError, ValueError)):
            domain_type.UniformDomain_2D(
                periodicity=(True, False),
                resolution=(8, 4),
                domain_bounds=((0.0, 1.0), ),  # type: ignore[arg-type]
            )


class TestProperties(unittest.TestCase):

    def test_lengths_widths_area_total_area(self):
        udomain_2d = domain_type.UniformDomain_2D(
            periodicity=(True, True),
            resolution=(10, 4),
            domain_bounds=((0.0, 2.0), (-1.0, 3.0)),
        )
        self.assertEqual(
            udomain_2d.domain_lengths,
            (2.0, 4.0),
        )
        self.assertEqual(
            udomain_2d.cell_widths,
            (0.2, 1.0),
        )
        self.assertAlmostEqual(
            udomain_2d.cell_area,
            0.2 * 1.0,
        )
        self.assertAlmostEqual(
            udomain_2d.total_area,
            2.0 * 4.0,
        )

    def test_cell_centers_shapes_and_values(self):
        udomain_2d = domain_type.UniformDomain_2D(
            periodicity=(True, True),
            resolution=(4, 2),
            domain_bounds=((0.0, 1.0), (0.0, 2.0)),
        )
        x0_centers, x1_centers = udomain_2d.cell_centers
        self.assertEqual(
            x0_centers.shape,
            (4, ),
        )
        self.assertEqual(
            x1_centers.shape,
            (2, ),
        )
        expected_x0 = numpy.array([0.125, 0.375, 0.625, 0.875])
        expected_x1 = numpy.array([0.5, 1.5])
        self.assertTrue(numpy.allclose(x0_centers, expected_x0))
        self.assertTrue(numpy.allclose(x1_centers, expected_x1))

    def test_cached_properties_return_same_object(self):
        udomain_2d = domain_type.UniformDomain_2D(
            periodicity=(True, True),
            resolution=(3, 3),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        )
        self.assertIs(
            udomain_2d.cell_widths,
            udomain_2d.cell_widths,
        )
        self.assertIs(
            udomain_2d.domain_lengths,
            udomain_2d.domain_lengths,
        )
        self.assertIs(
            udomain_2d.cell_centers,
            udomain_2d.cell_centers,
        )


class TestSliced3D(unittest.TestCase):

    def test_constructs_and_normalises_axis_enum(self):
        udomain_2d = domain_type.UniformDomain_2D_Sliced3D.from_slice(
            periodicity=(True, True),
            resolution=(8, 8),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            out_of_plane_axis="X1",
            slice_index=3,
            slice_position=0.5,
        )
        self.assertIs(
            udomain_2d.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X1,
        )
        self.assertEqual(
            udomain_2d.sliced_axis_index,
            1,
        )

    def test_accepts_int_and_enum_axis_inputs(self):
        udomain_from_int = domain_type.UniformDomain_2D_Sliced3D.from_slice(
            periodicity=(True, True),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            out_of_plane_axis=2,
            slice_index=0,
            slice_position=0.0,
        )
        self.assertIs(
            udomain_from_int.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X2,
        )
        self.assertEqual(
            udomain_from_int.sliced_axis_index,
            2,
        )

        udomain_from_enum = domain_type.UniformDomain_2D_Sliced3D.from_slice(
            periodicity=(True, True),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X0,
            slice_index=1,
            slice_position=0.25,
        )
        self.assertIs(
            udomain_from_enum.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertEqual(
            udomain_from_enum.sliced_axis_index,
            0,
        )

    def test_rejects_invalid_axis(self):
        with self.assertRaises(ValueError):
            domain_type.UniformDomain_2D_Sliced3D.from_slice(
                periodicity=(True, True),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
                out_of_plane_axis="x3",
                slice_index=0,
                slice_position=0.0,
            )

    def test_slice_index_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            domain_type.UniformDomain_2D_Sliced3D.from_slice(
                periodicity=(True, True),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
                out_of_plane_axis=0,
                slice_index=-1,
                slice_position=0.0,
            )

    def test_direct_init_rejects_negative_slice_index(self):
        with self.assertRaises(ValueError):
            domain_type.UniformDomain_2D_Sliced3D(
                periodicity=(True, True),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
                out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X0,
                slice_index=-1,
                slice_position=0.0,
            )

    def test_slice_position_must_be_finite(self):
        with self.assertRaises((TypeError, ValueError)):
            domain_type.UniformDomain_2D_Sliced3D.from_slice(
                periodicity=(True, True),
                resolution=(4, 4),
                domain_bounds=((0.0, 1.0), (0.0, 1.0)),
                out_of_plane_axis=0,
                slice_index=0,
                slice_position="a",  # type: ignore[arg-type]
            )


class TestEnsureHelpers(unittest.TestCase):

    def test_ensure_2d_udomain_accepts_subtype(self):
        udomain_2d = domain_type.UniformDomain_2D_Sliced3D.from_slice(
            periodicity=(True, True),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
            out_of_plane_axis=0,
            slice_index=0,
            slice_position=0.0,
        )
        domain_type.ensure_2d_udomain(udomain_2d=udomain_2d)

    def test_ensure_2d_udomain_sliced_from_3d_rejects_plain_2d(self):
        udomain_2d = domain_type.UniformDomain_2D(
            periodicity=(True, True),
            resolution=(4, 4),
            domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        )
        with self.assertRaises(TypeError):
            domain_type.ensure_2d_udomain_sliced_from_3d(udomain_2d=udomain_2d)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
