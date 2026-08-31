## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import Any

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import filter_farrays as _array_filter_farrays
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
    filter_fields,
)

##
## === HELPERS
##


def _make_3d_uniform_domain(
    resolution: tuple[int, int, int],
) -> domain_models.UniformDomain_3D:
    return domain_models.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )


def _make_sfield(
    sarray_3d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_models.ScalarField_3D:
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray_3d,
        uniform_domain_3d=_make_3d_uniform_domain(sarray_3d.shape),
        field_name="q",
        latex_label="q",
        sim_time=1.5,
    )


def _make_vfield(
    varray_3d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_models.VectorField_3D:
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray_3d,
        uniform_domain_3d=_make_3d_uniform_domain(varray_3d.shape[1:]),
        field_name="v",
        latex_label="v",
    )


def _make_r2tfield(
    r2tarray_3d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_models.RankTwoTensorField_3D:
    return field_models.RankTwoTensorField_3D.from_3d_r2tarray(
        r2tarray_3d=r2tarray_3d,
        uniform_domain_3d=_make_3d_uniform_domain(r2tarray_3d.shape[2:]),
        field_name="grad_v",
        latex_label="grad_v",
    )


##
## === TEST SUITES
##


class TestFieldMatchesArrayLevelKernel(unittest.TestCase):
    """Confirm the field-level path (extracting fdata.farray + uniform_domain.resolution)
    matches the array-level kernel exactly, for scalar, vector, and rank-2 tensor fields."""

    def test_sfield_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(0)
        sarray_3d = rng.standard_normal(resolution_3d)
        sfield = _make_sfield(sarray_3d)
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            sfield,
            k_min=1.0,
            k_max=2.0,
        )
        filtered_array = _array_filter_farrays.compute_bandpass_filtered_sarray(
            sarray_3d=sarray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(
            filtered_field.fdata.farray,
            filtered_array,
        )

    def test_vfield_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(1)
        varray_3d = rng.standard_normal((3, *resolution_3d))
        vfield = _make_vfield(varray_3d)
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            vfield,
            k_min=1.0,
            k_max=2.0,
        )
        filtered_array = _array_filter_farrays.compute_bandpass_filtered_varray(
            varray_3d=varray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(
            filtered_field.fdata.farray,
            filtered_array,
        )

    def test_r2tfield_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_3d = (num_cells, num_cells, num_cells)
        rng = numpy.random.default_rng(2)
        r2tarray_3d = rng.standard_normal((3, 3, *resolution_3d))
        r2tfield = _make_r2tfield(r2tarray_3d)
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            r2tfield,
            k_min=1.0,
            k_max=2.0,
        )
        filtered_array = _array_filter_farrays.compute_bandpass_filtered_r2tarray(
            r2tarray_3d=r2tarray_3d,
            resolution_3d=resolution_3d,
            k_min=1.0,
            k_max=2.0,
        )
        numpy.testing.assert_allclose(
            filtered_field.fdata.farray,
            filtered_array,
        )


class TestFieldIdentityIsPreserved(unittest.TestCase):
    """Filtering must not silently change what the field claims to be."""

    def test_default_field_name_and_label_are_preserved(
        self,
    ) -> None:
        sfield = _make_sfield(numpy.ones((8, 8, 8)))
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            sfield,
            k_min=1.0,
            k_max=2.0,
        )
        self.assertEqual(filtered_field.field_name, sfield.field_name)
        self.assertEqual(filtered_field.latex_label, sfield.latex_label)

    def test_field_name_and_label_can_be_overridden(
        self,
    ) -> None:
        sfield = _make_sfield(numpy.ones((8, 8, 8)))
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            sfield,
            k_min=1.0,
            k_max=2.0,
            field_name="q_filtered",
            latex_label="q_{\\mathrm{filtered}}",
        )
        self.assertEqual(filtered_field.field_name, "q_filtered")
        self.assertEqual(filtered_field.latex_label, "q_{\\mathrm{filtered}}")

    def test_uniform_domain_and_sim_time_are_preserved(
        self,
    ) -> None:
        sfield = _make_sfield(numpy.ones((8, 8, 8)))
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            sfield,
            k_min=1.0,
            k_max=2.0,
        )
        self.assertEqual(filtered_field.uniform_domain, sfield.uniform_domain)
        self.assertEqual(filtered_field.sim_time, sfield.sim_time)

    def test_output_is_same_concrete_type(
        self,
    ) -> None:
        vfield = _make_vfield(numpy.ones((3, 8, 8, 8)))
        filtered_field = filter_fields.compute_bandpass_filtered_field(
            vfield,
            k_min=1.0,
            k_max=2.0,
        )
        self.assertIsInstance(filtered_field, field_models.VectorField_3D)


class TestFieldTypeValidation(unittest.TestCase):
    """compute_bandpass_filtered_field must reject anything that isn't a
    supported 3D field type, rather than silently misinterpreting it."""

    def test_accepts_sfield(
        self,
    ) -> None:
        sfield = _make_sfield(numpy.ones((8, 8, 8)))
        filter_fields.compute_bandpass_filtered_field(sfield, k_min=1.0, k_max=2.0)

    def test_accepts_vfield(
        self,
    ) -> None:
        vfield = _make_vfield(numpy.ones((3, 8, 8, 8)))
        filter_fields.compute_bandpass_filtered_field(vfield, k_min=1.0, k_max=2.0)

    def test_accepts_r2tfield(
        self,
    ) -> None:
        r2tfield = _make_r2tfield(numpy.ones((3, 3, 8, 8, 8)))
        filter_fields.compute_bandpass_filtered_field(r2tfield, k_min=1.0, k_max=2.0)

    def test_rejects_bare_array(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            filter_fields.compute_bandpass_filtered_field(
                numpy.ones((8, 8, 8)),  # pyright: ignore[reportArgumentType]
                k_min=1.0,
                k_max=2.0,
            )

    def test_rejects_none(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            filter_fields.compute_bandpass_filtered_field(
                None,  # pyright: ignore[reportArgumentType]
                k_min=1.0,
                k_max=2.0,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
