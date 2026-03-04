## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest
import dataclasses

from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_2d import domain_type, field_type

##
## === HELPERS
##


def _make_2d_udomain(
    resolution: tuple = (4, 4),
) -> domain_type.UniformDomain_2D:
    return domain_type.UniformDomain_2D(
        periodicity=(True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
    )


def _make_sliced_domain_2d(
    resolution: tuple = (4, 4),
    out_of_plane_axis: int = 2,
    slice_index: int = 0,
    slice_position: float = 0.125,
) -> domain_type.UniformDomain_2D_Sliced3D:
    return domain_type.UniformDomain_2D_Sliced3D.from_slice(
        periodicity=(True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        out_of_plane_axis=out_of_plane_axis,
        slice_index=slice_index,
        slice_position=slice_position,
    )


def _make_sfield_2d(
    resolution: tuple = (4, 4),
    label: str = "test_scalar_2d",
    sim_time: float | None = None,
    use_sliced_domain: bool = False,
) -> field_type.ScalarField_2D:
    udomain = _make_sliced_domain_2d(resolution) if use_sliced_domain else _make_2d_udomain(resolution)
    return field_type.ScalarField_2D.from_2d_sarray(
        sarray_2d=numpy.ones(resolution),
        udomain_2d=udomain,
        field_label=label,
        sim_time=sim_time,
    )


def _make_vfield_2d(
    resolution: tuple = (4, 4),
    label: str = "test_vector_2d",
    sim_time: float | None = None,
    use_sliced_domain: bool = False,
) -> field_type.VectorField_2D:
    udomain = _make_sliced_domain_2d(resolution) if use_sliced_domain else _make_2d_udomain(resolution)
    return field_type.VectorField_2D.from_2d_varray(
        varray_2d=numpy.ones((2, ) + resolution),
        udomain_2d=udomain,
        field_label=label,
        sim_time=sim_time,
    )


##
## === TEST SUITES
##


class TestScalarField2D_Construction(unittest.TestCase):

    def test_valid_construction_via_from_2d_sarray(self):
        sfield = _make_sfield_2d()
        self.assertIsInstance(
            sfield,
            field_type.ScalarField_2D,
        )

    def test_label_is_stored(self):
        sfield = _make_sfield_2d(label="temperature")
        self.assertEqual(
            sfield.field_label,
            "temperature",
        )

    def test_sim_time_is_stored(self):
        sfield = _make_sfield_2d(sim_time=1.5)
        self.assertAlmostEqual(
            sfield.sim_time,
            1.5,  # type: ignore[arg-type]
        )

    def test_sim_time_none_allowed(self):
        sfield = _make_sfield_2d(sim_time=None)
        self.assertIsNone(sfield.sim_time)

    def test_udomain_is_stored(self):
        domain = _make_2d_udomain()
        sfield = field_type.ScalarField_2D.from_2d_sarray(
            sarray_2d=numpy.ones((4, 4)),
            udomain_2d=domain,
            field_label="f",
        )
        self.assertEqual(
            sfield.udomain,
            domain,
        )

    def test_empty_label_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_2D.from_2d_sarray(
                sarray_2d=numpy.ones((4, 4)),
                udomain_2d=_make_2d_udomain(),
                field_label="",
            )

    def test_wrong_array_rank_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_2D.from_2d_sarray(
                sarray_2d=numpy.ones((4,)),  # type: ignore[arg-type]
                udomain_2d=_make_2d_udomain(),
                field_label="bad",
            )

    def test_3d_array_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_2D.from_2d_sarray(
                sarray_2d=numpy.ones((4, 4, 4)),  # type: ignore[arg-type]
                udomain_2d=_make_2d_udomain(),
                field_label="bad",
            )

    def test_resolution_mismatch_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.ScalarField_2D.from_2d_sarray(
                sarray_2d=numpy.ones((4, 4)),
                udomain_2d=_make_2d_udomain(resolution=(8, 8)),
                field_label="bad",
            )

    def test_frozen_immutability(self):
        sfield = _make_sfield_2d()
        with self.assertRaises((
                dataclasses.FrozenInstanceError,
                AttributeError,
                TypeError,
        )):
            sfield.field_label = "modified"  # type: ignore[misc]


class TestScalarField2D_Properties(unittest.TestCase):

    def test_fdata_is_scalar(self):
        sfield = _make_sfield_2d()
        self.assertTrue(sfield.fdata.is_scalar)
        self.assertFalse(sfield.fdata.is_vector)
        self.assertFalse(sfield.fdata.is_tensor)

    def test_fdata_num_ranks(self):
        sfield = _make_sfield_2d()
        self.assertEqual(
            sfield.fdata.num_ranks,
            0,
        )

    def test_fdata_num_comps(self):
        sfield = _make_sfield_2d()
        self.assertEqual(
            sfield.fdata.num_comps,
            1,
        )

    def test_fdata_num_sdims(self):
        sfield = _make_sfield_2d()
        self.assertEqual(
            sfield.fdata.num_sdims,
            2,
        )

    def test_fdata_sdims_shape(self):
        sfield = _make_sfield_2d(resolution=(3, 7))
        self.assertEqual(
            sfield.fdata.sdims_shape,
            (3, 7),
        )

    def test_fdata_comps_shape_is_empty_for_scalar(self):
        sfield = _make_sfield_2d()
        self.assertEqual(
            sfield.fdata.comps_shape,
            (),
        )

    def test_fdata_shape_equals_resolution_for_scalar(self):
        sfield = _make_sfield_2d(resolution=(3, 7))
        self.assertEqual(
            sfield.fdata.shape,
            (3, 7),
        )

    def test_fdata_array_values_preserved(self):
        sarray = numpy.arange(12, dtype=float).reshape((3, 4))
        domain = _make_2d_udomain(resolution=(3, 4))
        sfield = field_type.ScalarField_2D.from_2d_sarray(
            sarray_2d=sarray,
            udomain_2d=domain,
            field_label="f",
        )
        self.assertTrue(
            numpy.array_equal(
                sfield.fdata.farray,
                sarray,
            ),
        )


class TestScalarField2D_IsSlicedFrom3D(unittest.TestCase):

    def test_plain_2d_domain_not_sliced(self):
        sfield = _make_sfield_2d(use_sliced_domain=False)
        self.assertFalse(sfield.is_sliced_from_3d)

    def test_sliced_3d_domain_is_detected(self):
        sfield = _make_sfield_2d(use_sliced_domain=True)
        self.assertTrue(sfield.is_sliced_from_3d)

    def test_sliced_domain_metadata_accessible(self):
        sliced_domain = _make_sliced_domain_2d(
            out_of_plane_axis=1,
            slice_index=2,
            slice_position=0.25,
        )
        sfield = field_type.ScalarField_2D.from_2d_sarray(
            sarray_2d=numpy.ones((4, 4)),
            udomain_2d=sliced_domain,
            field_label="f",
        )
        self.assertTrue(sfield.is_sliced_from_3d)
        self.assertIsInstance(
            sfield.udomain,
            domain_type.UniformDomain_2D_Sliced3D,
        )
        sliced_3d_domain = sfield.udomain
        assert isinstance(
            sliced_3d_domain,
            domain_type.UniformDomain_2D_Sliced3D,
        )
        self.assertIs(
            sliced_3d_domain.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X1,
        )
        self.assertEqual(
            sliced_3d_domain.slice_index,
            2,
        )
        self.assertAlmostEqual(
            sliced_3d_domain.slice_position,
            0.25,
        )


class TestVectorField2D_Construction(unittest.TestCase):

    def test_valid_construction_via_from_2d_varray(self):
        vfield = _make_vfield_2d()
        self.assertIsInstance(
            vfield,
            field_type.VectorField_2D,
        )

    def test_label_is_stored(self):
        vfield = _make_vfield_2d(label="velocity_2d")
        self.assertEqual(
            vfield.field_label,
            "velocity_2d",
        )

    def test_sim_time_is_stored(self):
        vfield = _make_vfield_2d(sim_time=0.5)
        self.assertAlmostEqual(
            vfield.sim_time,
            0.5,  # type: ignore[arg-type]
        )

    def test_wrong_leading_dim_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_2D.from_2d_varray(
                varray_2d=numpy.ones((3, 4, 4)),  # type: ignore[arg-type]
                udomain_2d=_make_2d_udomain(),
                field_label="bad",
            )

    def test_wrong_array_rank_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_2D.from_2d_varray(
                varray_2d=numpy.ones((2, 4)),  # type: ignore[arg-type]
                udomain_2d=_make_2d_udomain(),
                field_label="bad",
            )

    def test_resolution_mismatch_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_2D.from_2d_varray(
                varray_2d=numpy.ones((2, 4, 4)),
                udomain_2d=_make_2d_udomain(resolution=(8, 8)),
                field_label="bad",
            )

    def test_empty_label_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            field_type.VectorField_2D.from_2d_varray(
                varray_2d=numpy.ones((2, 4, 4)),
                udomain_2d=_make_2d_udomain(),
                field_label="",
            )

    def test_frozen_immutability(self):
        vfield = _make_vfield_2d()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            vfield.field_label = "modified"  # type: ignore[misc]


class TestVectorField2D_Properties(unittest.TestCase):

    def test_fdata_is_vector(self):
        vfield = _make_vfield_2d()
        self.assertTrue(vfield.fdata.is_vector)
        self.assertFalse(vfield.fdata.is_scalar)
        self.assertFalse(vfield.fdata.is_tensor)

    def test_fdata_num_ranks(self):
        vfield = _make_vfield_2d()
        self.assertEqual(
            vfield.fdata.num_ranks,
            1,
        )

    def test_fdata_num_comps(self):
        vfield = _make_vfield_2d()
        self.assertEqual(
            vfield.fdata.num_comps,
            2,
        )

    def test_fdata_num_sdims(self):
        vfield = _make_vfield_2d()
        self.assertEqual(
            vfield.fdata.num_sdims,
            2,
        )

    def test_fdata_sdims_shape(self):
        vfield = _make_vfield_2d(resolution=(3, 7))
        self.assertEqual(
            vfield.fdata.sdims_shape,
            (3, 7),
        )

    def test_fdata_comps_shape(self):
        vfield = _make_vfield_2d()
        self.assertEqual(
            vfield.fdata.comps_shape,
            (2, ),
        )

    def test_fdata_shape_includes_component_axis(self):
        vfield = _make_vfield_2d(resolution=(3, 7))
        self.assertEqual(
            vfield.fdata.shape,
            (2, 3, 7),
        )

    def test_fdata_array_values_preserved(self):
        varray = numpy.arange(24, dtype=float).reshape((2, 3, 4))
        domain = _make_2d_udomain(resolution=(3, 4))
        vfield = field_type.VectorField_2D.from_2d_varray(
            varray_2d=varray,
            udomain_2d=domain,
            field_label="v",
        )
        self.assertTrue(
            numpy.array_equal(
                vfield.fdata.farray,
                varray,
            ),
        )


class TestVectorField2D_IsSlicedFrom3D(unittest.TestCase):

    def test_plain_2d_domain_not_sliced(self):
        vfield = _make_vfield_2d(use_sliced_domain=False)
        self.assertFalse(vfield.is_sliced_from_3d)

    def test_sliced_3d_domain_is_detected(self):
        vfield = _make_vfield_2d(use_sliced_domain=True)
        self.assertTrue(vfield.is_sliced_from_3d)

    def test_sliced_domain_metadata_accessible(self):
        sliced_domain = _make_sliced_domain_2d(
            out_of_plane_axis=0,
            slice_index=3,
            slice_position=0.75,
        )
        vfield = field_type.VectorField_2D.from_2d_varray(
            varray_2d=numpy.ones((2, 4, 4)),
            udomain_2d=sliced_domain,
            field_label="v",
        )
        self.assertTrue(vfield.is_sliced_from_3d)
        self.assertIsInstance(
            vfield.udomain,
            domain_type.UniformDomain_2D_Sliced3D,
        )
        sliced_3d_domain = vfield.udomain
        assert isinstance(
            sliced_3d_domain,
            domain_type.UniformDomain_2D_Sliced3D,
        )
        self.assertIs(
            sliced_3d_domain.out_of_plane_axis,
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertEqual(
            sliced_3d_domain.slice_index,
            3,
        )
        self.assertAlmostEqual(
            sliced_3d_domain.slice_position,
            0.75,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
