## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest

from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_2d import (
    field_types as field_types_2d,
)
from jormi.ww_fields.fields_3d import (
    domain_types as domain_types_3d,
    field_types as field_types_3d,
    slice_fields,
)

##
## === TEST SUITES
##


class TestUdomain(unittest.TestCase):

    def _make_udomain_3d(
        self,
    ) -> domain_types_3d.UniformDomain_3D:
        return domain_types_3d.UniformDomain_3D(
            periodicity=(True, False, True),
            resolution=(3, 4, 5),
            domain_bounds=((0.0, 3.0), (0.0, 4.0), (0.0, 5.0)),
        )

    def test_constructs_sliced_domain_axis_x0(self):
        udomain_3d = self._make_udomain_3d()
        udomain_2d = slice_fields._slice_3d_udomain(
            udomain_3d=udomain_3d,
            out_of_plane_axis="x0",
            slice_index=1,
            param_name="<udomain_3d>",
        )
        self.assertEqual(
            udomain_2d.num_sdims,
            2,
        )
        self.assertIs(
            udomain_2d.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertEqual(
            udomain_2d.periodicity,
            (False, True),
        )
        self.assertEqual(
            udomain_2d.resolution,
            (4, 5),
        )
        self.assertEqual(
            udomain_2d.domain_bounds,
            ((0.0, 4.0), (0.0, 5.0)),
        )
        self.assertEqual(
            udomain_2d.slice_index,
            1,
        )
        self.assertAlmostEqual(
            udomain_2d.slice_position,
            1.5,
        )

    def test_constructs_sliced_domain_axis_x1(self):
        udomain_3d = self._make_udomain_3d()
        udomain_2d = slice_fields._slice_3d_udomain(
            udomain_3d=udomain_3d,
            out_of_plane_axis=1,
            slice_index=2,
            param_name="<udomain_3d>",
        )
        self.assertIs(
            udomain_2d.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X1,
        )
        self.assertEqual(
            udomain_2d.periodicity,
            (True, True),
        )
        self.assertEqual(
            udomain_2d.resolution,
            (3, 5),
        )
        self.assertEqual(
            udomain_2d.domain_bounds,
            ((0.0, 3.0), (0.0, 5.0)),
        )
        self.assertEqual(
            udomain_2d.slice_index,
            2,
        )
        self.assertAlmostEqual(
            udomain_2d.slice_position,
            2.5,
        )

    def test_constructs_sliced_domain_axis_x2(self):
        udomain_3d = self._make_udomain_3d()
        udomain_2d = slice_fields._slice_3d_udomain(
            udomain_3d=udomain_3d,
            out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X2,
            slice_index=4,
            param_name="<udomain_3d>",
        )
        self.assertIs(
            udomain_2d.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X2,
        )
        self.assertEqual(
            udomain_2d.periodicity,
            (True, False),
        )
        self.assertEqual(
            udomain_2d.resolution,
            (3, 4),
        )
        self.assertEqual(
            udomain_2d.domain_bounds,
            ((0.0, 3.0), (0.0, 4.0)),
        )
        self.assertEqual(
            udomain_2d.slice_index,
            4,
        )
        self.assertAlmostEqual(
            udomain_2d.slice_position,
            4.5,
        )

    def test_rejects_negative_slice_index(self):
        udomain_3d = self._make_udomain_3d()
        with self.assertRaises(ValueError):
            slice_fields._slice_3d_udomain(
                udomain_3d=udomain_3d,
                out_of_plane_axis="x0",
                slice_index=-1,
                param_name="<udomain_3d>",
            )

    def test_rejects_slice_index_out_of_range(self):
        udomain_3d = self._make_udomain_3d()
        with self.assertRaises(ValueError):
            slice_fields._slice_3d_udomain(
                udomain_3d=udomain_3d,
                out_of_plane_axis="X2",
                slice_index=5,
                param_name="<udomain_3d>",
            )

    def test_rejects_invalid_axis(self):
        udomain_3d = self._make_udomain_3d()
        with self.assertRaises(ValueError):
            slice_fields._slice_3d_udomain(
                udomain_3d=udomain_3d,
                out_of_plane_axis="x3",
                slice_index=0,
                param_name="<udomain_3d>",
            )


class TestScalarField(unittest.TestCase):

    def _make_udomain_3d(
        self,
    ) -> domain_types_3d.UniformDomain_3D:
        return domain_types_3d.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(3, 4, 5),
            domain_bounds=((0.0, 3.0), (0.0, 4.0), (0.0, 5.0)),
        )

    def _make_sfield_3d(
        self,
    ) -> field_types_3d.ScalarField_3D:
        udomain_3d = self._make_udomain_3d()
        sarray_3d = numpy.arange(3 * 4 * 5, dtype=float).reshape((3, 4, 5))
        return field_types_3d.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            udomain_3d=udomain_3d,
            field_label="rho",
            sim_time=1.25,
        )

    def test_slices_axis_x0(self):
        sfield_3d = self._make_sfield_3d()
        sfield_2d = slice_fields.slice_3d_sfield(
            sfield_3d=sfield_3d,
            out_of_plane_axis=0,
            slice_index=1,
        )
        sarray_3d = field_types_3d.extract_3d_sarray(sfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (4, 5),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                sarray_3d[1, :, :],
            ),
        )
        self.assertEqual(
            sfield_2d.sim_time,
            sfield_3d.sim_time,
        )
        self.assertEqual(
            sfield_2d.field_label,
            sfield_3d.field_label,
        )

    def test_slices_axis_x1(self):
        sfield_3d = self._make_sfield_3d()
        sfield_2d = slice_fields.slice_3d_sfield(
            sfield_3d=sfield_3d,
            out_of_plane_axis="X1",
            slice_index=2,
        )
        sarray_3d = field_types_3d.extract_3d_sarray(sfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (3, 5),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                sarray_3d[:, 2, :],
            ),
        )

    def test_slices_axis_x2(self):
        sfield_3d = self._make_sfield_3d()
        sfield_2d = slice_fields.slice_3d_sfield(
            sfield_3d=sfield_3d,
            out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X2,
            slice_index=4,
        )
        sarray_3d = field_types_3d.extract_3d_sarray(sfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (3, 4),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                sarray_3d[:, :, 4],
            ),
        )

    def test_rejects_negative_slice_index(self):
        sfield_3d = self._make_sfield_3d()
        with self.assertRaises(ValueError):
            slice_fields.slice_3d_sfield(
                sfield_3d=sfield_3d,
                out_of_plane_axis="x0",
                slice_index=-1,
            )


class TestVectorField(unittest.TestCase):

    def _make_udomain_3d(
        self,
    ) -> domain_types_3d.UniformDomain_3D:
        return domain_types_3d.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(3, 4, 5),
            domain_bounds=((0.0, 3.0), (0.0, 4.0), (0.0, 5.0)),
        )

    def _make_vfield_3d(
        self,
    ) -> field_types_3d.VectorField_3D:
        udomain_3d = self._make_udomain_3d()
        varray_3d = numpy.arange(3 * 3 * 4 * 5, dtype=float).reshape((3, 3, 4, 5))
        return field_types_3d.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            udomain_3d=udomain_3d,
            field_label="u",
            sim_time=2.0,
        )

    def test_slices_inplane_axis_x0(self):
        vfield_3d = self._make_vfield_3d()
        vfield_2d = slice_fields.slice_3d_vfield_inplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis=0,
            slice_index=1,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        varray_2d = field_types_2d.extract_2d_varray(vfield_2d)
        self.assertEqual(
            varray_2d.shape,
            (2, 4, 5),
        )
        expected = varray_3d[[1, 2], 1, :, :]
        self.assertTrue(
            numpy.allclose(
                varray_2d,
                expected,
            ),
        )

    def test_slices_inplane_axis_x1(self):
        vfield_3d = self._make_vfield_3d()
        vfield_2d = slice_fields.slice_3d_vfield_inplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis="X1",
            slice_index=2,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        varray_2d = field_types_2d.extract_2d_varray(vfield_2d)
        self.assertEqual(
            varray_2d.shape,
            (2, 3, 5),
        )
        expected = varray_3d[[0, 2], :, 2, :]
        self.assertTrue(
            numpy.allclose(
                varray_2d,
                expected,
            ),
        )

    def test_slices_inplane_axis_x2(self):
        vfield_3d = self._make_vfield_3d()
        vfield_2d = slice_fields.slice_3d_vfield_inplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X2,
            slice_index=4,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        varray_2d = field_types_2d.extract_2d_varray(vfield_2d)
        self.assertEqual(
            varray_2d.shape,
            (2, 3, 4),
        )
        expected = varray_3d[[0, 1], :, :, 4]
        self.assertTrue(
            numpy.allclose(
                varray_2d,
                expected,
            ),
        )

    def test_slices_outofplane_axis_x0(self):
        vfield_3d = self._make_vfield_3d()
        sfield_2d = slice_fields.slice_3d_vfield_outofplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis=0,
            slice_index=1,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (4, 5),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                varray_3d[0, 1, :, :],
            ),
        )

    def test_slices_outofplane_axis_x1(self):
        vfield_3d = self._make_vfield_3d()
        sfield_2d = slice_fields.slice_3d_vfield_outofplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis="X1",
            slice_index=2,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (3, 5),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                varray_3d[1, :, 2, :],
            ),
        )

    def test_slices_outofplane_axis_x2(self):
        vfield_3d = self._make_vfield_3d()
        sfield_2d = slice_fields.slice_3d_vfield_outofplane(
            vfield_3d=vfield_3d,
            out_of_plane_axis=cartesian_axes.CartesianAxis_3D.X2,
            slice_index=4,
        )
        varray_3d = field_types_3d.extract_3d_varray(vfield_3d)
        sarray_2d = field_types_2d.extract_2d_sarray(sfield_2d)
        self.assertEqual(
            sarray_2d.shape,
            (3, 4),
        )
        self.assertTrue(
            numpy.allclose(
                sarray_2d,
                varray_3d[2, :, :, 4],
            ),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
