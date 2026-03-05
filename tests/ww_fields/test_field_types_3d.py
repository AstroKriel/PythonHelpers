## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest
import dataclasses

from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_type, field_type

##
## === HELPERS
##


def _make_3d_udomain(
    resolution: tuple = (4, 4, 4),
) -> domain_type.UniformDomain_3D:
    return domain_type.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )


def _make_sfield_3d(
    resolution: tuple = (4, 4, 4),
    label: str = "test_scalar",
    sim_time: float | None = None,
) -> field_type.ScalarField_3D:
    return field_type.ScalarField_3D.from_3d_sarray(
        sarray_3d=numpy.ones(resolution),
        udomain_3d=_make_3d_udomain(resolution),
        field_label=label,
        sim_time=sim_time,
    )


def _make_vfield_3d(
    resolution: tuple = (4, 4, 4),
    label: str = "test_vector",
    sim_time: float | None = None,
) -> field_type.VectorField_3D:
    return field_type.VectorField_3D.from_3d_varray(
        varray_3d=numpy.ones((3, ) + resolution),
        udomain_3d=_make_3d_udomain(resolution),
        field_label=label,
        sim_time=sim_time,
    )


def _make_unit_varray_3d(
    resolution: tuple,
) -> numpy.ndarray:
    """Return a (3, Nx, Ny, Nz) array with unit vectors pointing in the x0 direction."""
    varray = numpy.zeros((3, ) + resolution)
    varray[0] = 1.0
    return varray


##
## === TEST SUITES
##


class TestScalarField3D_Construction(unittest.TestCase):

    def test_valid_construction_via_from_3d_sarray(self):
        sfield = _make_sfield_3d()
        self.assertIsInstance(
            sfield,
            field_type.ScalarField_3D,
        )

    def test_label_is_stored(self):
        sfield = _make_sfield_3d(label="pressure")
        self.assertEqual(
            sfield.field_label,
            "pressure",
        )

    def test_sim_time_is_stored(self):
        sfield = _make_sfield_3d(sim_time=1.5)
        self.assertAlmostEqual(
            sfield.sim_time,
            1.5,  # type: ignore[arg-type]
        )

    def test_sim_time_none_allowed(self):
        sfield = _make_sfield_3d(sim_time=None)
        self.assertIsNone(sfield.sim_time)

    def test_udomain_is_stored(self):
        domain = _make_3d_udomain()
        sfield = field_type.ScalarField_3D.from_3d_sarray(
            sarray_3d=numpy.ones((4, 4, 4)),
            udomain_3d=domain,
            field_label="f",
        )
        self.assertEqual(
            sfield.udomain,
            domain,
        )

    def test_empty_label_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_3D.from_3d_sarray(
                sarray_3d=numpy.ones((4, 4, 4)),
                udomain_3d=_make_3d_udomain(),
                field_label="",
            )

    def test_wrong_array_rank_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_3D.from_3d_sarray(
                sarray_3d=numpy.ones((4, 4)),  # type: ignore[arg-type]
                udomain_3d=_make_3d_udomain(),
                field_label="bad",
            )

    def test_resolution_mismatch_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_3D.from_3d_sarray(
                sarray_3d=numpy.ones((4, 4, 4)),
                udomain_3d=_make_3d_udomain(resolution=(8, 8, 8)),
                field_label="bad",
            )

    def test_frozen_immutability(self):
        sfield = _make_sfield_3d()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            sfield.field_label = "modified"  # type: ignore[misc]


class TestScalarField3D_Properties(unittest.TestCase):

    def test_fdata_is_scalar(self):
        sfield = _make_sfield_3d()
        self.assertTrue(sfield.fdata.is_scalar)
        self.assertFalse(sfield.fdata.is_vector)
        self.assertFalse(sfield.fdata.is_tensor)

    def test_fdata_num_ranks(self):
        sfield = _make_sfield_3d()
        self.assertEqual(
            sfield.fdata.num_ranks,
            0,
        )

    def test_fdata_num_comps(self):
        sfield = _make_sfield_3d()
        self.assertEqual(
            sfield.fdata.num_comps,
            1,
        )

    def test_fdata_num_sdims(self):
        sfield = _make_sfield_3d()
        self.assertEqual(
            sfield.fdata.num_sdims,
            3,
        )

    def test_fdata_sdims_shape(self):
        sfield = _make_sfield_3d(resolution=(3, 5, 7))
        self.assertEqual(
            sfield.fdata.sdims_shape,
            (3, 5, 7),
        )

    def test_fdata_comps_shape_is_empty_for_scalar(self):
        sfield = _make_sfield_3d()
        self.assertEqual(
            sfield.fdata.comps_shape,
            (),
        )

    def test_fdata_shape_equals_resolution_for_scalar(self):
        sfield = _make_sfield_3d(resolution=(3, 5, 7))
        self.assertEqual(
            sfield.fdata.shape,
            (3, 5, 7),
        )

    def test_fdata_array_values_preserved(self):
        sarray = numpy.arange(24, dtype=float).reshape((2, 3, 4))
        domain = _make_3d_udomain(resolution=(2, 3, 4))
        sfield = field_type.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray,
            udomain_3d=domain,
            field_label="f",
        )
        self.assertTrue(
            numpy.array_equal(
                sfield.fdata.farray,
                sarray,
            ),
        )


class TestVectorField3D_Construction(unittest.TestCase):

    def test_valid_construction_via_from_3d_varray(self):
        vfield = _make_vfield_3d()
        self.assertIsInstance(
            vfield,
            field_type.VectorField_3D,
        )

    def test_label_is_stored(self):
        vfield = _make_vfield_3d(label="velocity")
        self.assertEqual(
            vfield.field_label,
            "velocity",
        )

    def test_sim_time_is_stored(self):
        vfield = _make_vfield_3d(sim_time=3.0)
        self.assertAlmostEqual(
            vfield.sim_time,
            3.0,  # type: ignore[arg-type]
        )

    def test_wrong_leading_dim_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_3D.from_3d_varray(
                varray_3d=numpy.ones((2, 4, 4, 4)),  # type: ignore[arg-type]
                udomain_3d=_make_3d_udomain(),
                field_label="bad",
            )

    def test_wrong_array_rank_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_3D.from_3d_varray(
                varray_3d=numpy.ones((3, 4, 4)),  # type: ignore[arg-type]
                udomain_3d=_make_3d_udomain(),
                field_label="bad",
            )

    def test_resolution_mismatch_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_3D.from_3d_varray(
                varray_3d=numpy.ones((3, 4, 4, 4)),
                udomain_3d=_make_3d_udomain(resolution=(8, 8, 8)),
                field_label="bad",
            )

    def test_empty_label_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_3D.from_3d_varray(
                varray_3d=numpy.ones((3, 4, 4, 4)),
                udomain_3d=_make_3d_udomain(),
                field_label="",
            )

    def test_frozen_immutability(self):
        vfield = _make_vfield_3d()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            vfield.field_label = "modified"  # type: ignore[misc]


class TestVectorField3D_Properties(unittest.TestCase):

    def test_fdata_is_vector(self):
        vfield = _make_vfield_3d()
        self.assertTrue(vfield.fdata.is_vector)
        self.assertFalse(vfield.fdata.is_scalar)
        self.assertFalse(vfield.fdata.is_tensor)

    def test_fdata_num_ranks(self):
        vfield = _make_vfield_3d()
        self.assertEqual(
            vfield.fdata.num_ranks,
            1,
        )

    def test_fdata_num_comps(self):
        vfield = _make_vfield_3d()
        self.assertEqual(
            vfield.fdata.num_comps,
            3,
        )

    def test_fdata_num_sdims(self):
        vfield = _make_vfield_3d()
        self.assertEqual(
            vfield.fdata.num_sdims,
            3,
        )

    def test_fdata_sdims_shape(self):
        vfield = _make_vfield_3d(resolution=(3, 5, 7))
        self.assertEqual(
            vfield.fdata.sdims_shape,
            (3, 5, 7),
        )

    def test_fdata_comps_shape(self):
        vfield = _make_vfield_3d()
        self.assertEqual(
            vfield.fdata.comps_shape,
            (3, ),
        )

    def test_fdata_shape_includes_component_axis(self):
        vfield = _make_vfield_3d(resolution=(3, 5, 7))
        self.assertEqual(
            vfield.fdata.shape,
            (3, 3, 5, 7),
        )

    def test_default_comp_axes(self):
        vfield = _make_vfield_3d()
        self.assertEqual(
            vfield.comp_axes,
            cartesian_axes.DEFAULT_3D_AXES_ORDER,
        )


class TestVectorField3D_GetVcomp(unittest.TestCase):

    def setUp(self):
        self._resolution = (4, 5, 6)
        rng = numpy.random.default_rng(0)
        self._varray = rng.standard_normal((3, ) + self._resolution)
        self._vfield = field_type.VectorField_3D.from_3d_varray(
            varray_3d=self._varray,
            udomain_3d=_make_3d_udomain(self._resolution),
            field_label="v",
        )

    def test_get_vcomp_x0_shape(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X0)
        self.assertEqual(
            sarray.shape,
            self._resolution,
        )

    def test_get_vcomp_x1_shape(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X1)
        self.assertEqual(
            sarray.shape,
            self._resolution,
        )

    def test_get_vcomp_x2_shape(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X2)
        self.assertEqual(
            sarray.shape,
            self._resolution,
        )

    def test_get_vcomp_x0_values(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X0)
        self.assertTrue(
            numpy.array_equal(
                sarray,
                self._varray[0],
            ),
        )

    def test_get_vcomp_x1_values(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X1)
        self.assertTrue(
            numpy.array_equal(
                sarray,
                self._varray[1],
            ),
        )

    def test_get_vcomp_x2_values(self):
        sarray = self._vfield.get_vcomp_3d_sarray(cartesian_axes.CartesianAxis_3D.X2)
        self.assertTrue(
            numpy.array_equal(
                sarray,
                self._varray[2],
            ),
        )

    def test_get_vcomp_by_integer_axis(self):
        sarray = self._vfield.get_vcomp_3d_sarray(0)
        self.assertTrue(
            numpy.array_equal(
                sarray,
                self._varray[0],
            ),
        )

    def test_get_vcomp_by_string_axis(self):
        sarray = self._vfield.get_vcomp_3d_sarray("x1")
        self.assertTrue(
            numpy.array_equal(
                sarray,
                self._varray[1],
            ),
        )


class TestUnitVectorField3D(unittest.TestCase):

    def _make_unit_vfield(
        self,
        resolution: tuple = (4, 4, 4),
    ) -> field_type.VectorField_3D:
        return field_type.VectorField_3D.from_3d_varray(
            varray_3d=_make_unit_varray_3d(resolution),
            udomain_3d=_make_3d_udomain(resolution),
            field_label="unit_v",
        )

    def test_valid_unit_vectors_accepted(self):
        vfield = self._make_unit_vfield()
        uvfield = field_type.UnitVectorField_3D.from_3d_vfield(vfield)
        self.assertIsInstance(
            uvfield,
            field_type.UnitVectorField_3D,
        )

    def test_non_unit_vectors_raise(self):
        vfield = _make_vfield_3d()  # magnitude = sqrt(3), not 1
        with self.assertRaises(ValueError):
            field_type.UnitVectorField_3D.from_3d_vfield(vfield)

    def test_as_3d_uvfield_function_accepts_unit_field(self):
        vfield = self._make_unit_vfield()
        uvfield = field_type.as_3d_uvfield(vfield)
        self.assertIsInstance(
            uvfield,
            field_type.UnitVectorField_3D,
        )

    def test_uvfield_is_subtype_of_vfield(self):
        vfield = self._make_unit_vfield()
        uvfield = field_type.UnitVectorField_3D.from_3d_vfield(vfield)
        self.assertIsInstance(
            uvfield,
            field_type.VectorField_3D,
        )

    def test_from_3d_vfield_preserves_label_and_sim_time(self):
        vfield = field_type.VectorField_3D.from_3d_varray(
            varray_3d=_make_unit_varray_3d((4, 4, 4)),
            udomain_3d=_make_3d_udomain(),
            field_label="my_label",
            sim_time=2.5,
        )
        uvfield = field_type.UnitVectorField_3D.from_3d_vfield(vfield)
        self.assertEqual(
            uvfield.field_label,
            "my_label",
        )
        self.assertAlmostEqual(
            uvfield.sim_time,
            2.5,  # type: ignore[arg-type]
        )
        self.assertEqual(
            uvfield.udomain,
            vfield.udomain,
        )

    def test_custom_tolerance_accepts_slightly_off_unit(self):
        varray = _make_unit_varray_3d((4, 4, 4))
        varray[0] *= 1.0001  # 0.01 percent deviation
        vfield = field_type.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=_make_3d_udomain(),
            field_label="approx_unit",
        )
        uvfield = field_type.UnitVectorField_3D.from_3d_vfield(
            vfield,
            tol=1e-2,
        )
        self.assertIsInstance(
            uvfield,
            field_type.UnitVectorField_3D,
        )

    def test_tight_tolerance_rejects_slightly_off_unit(self):
        varray = _make_unit_varray_3d((4, 4, 4))
        varray[0] *= 1.01  # 1 percent deviation
        vfield = field_type.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=_make_3d_udomain(),
            field_label="approx_unit",
        )
        with self.assertRaises(ValueError):
            field_type.UnitVectorField_3D.from_3d_vfield(
                vfield,
                tol=1e-6,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
