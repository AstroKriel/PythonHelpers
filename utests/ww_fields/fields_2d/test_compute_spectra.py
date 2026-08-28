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
from jormi.ww_arrays.farrays_2d import compute_spectra as _array_compute_spectra
from jormi.ww_fields.fields_2d import (
    compute_spectra,
    domain_models,
    field_models,
)

##
## === HELPERS
##


def _make_2d_uniform_domain(
    resolution: tuple[int, int],
) -> domain_models.UniformDomain_2D:
    return domain_models.UniformDomain_2D(
        periodicity=(True, True),
        resolution=resolution,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
    )


def _make_sfield(
    sarray_2d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_models.ScalarField_2D:
    return field_models.ScalarField_2D.from_2d_sarray(
        sarray_2d=sarray_2d,
        uniform_domain_2d=_make_2d_uniform_domain(sarray_2d.shape),
        field_name="q",
        latex_label="q",
    )


def _make_vfield(
    varray_2d: numpy.ndarray[Any, numpy.dtype[Any]],
) -> field_models.VectorField_2D:
    return field_models.VectorField_2D.from_2d_varray(
        varray_2d=varray_2d,
        uniform_domain_2d=_make_2d_uniform_domain(varray_2d.shape[1:]),
        field_name="v",
        latex_label="v",
    )


##
## === TEST SUITES
##


class TestFieldMatchesArrayLevelKernel(unittest.TestCase):
    """Confirm the field-level path (extracting fdata.farray + uniform_domain.resolution)
    matches the array-level kernel exactly, for both scalar and vector fields."""

    def test_sfield_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_2d = (num_cells, num_cells)
        rng = numpy.random.default_rng(0)
        sarray_2d = rng.standard_normal(resolution_2d)
        sfield = _make_sfield(sarray_2d)
        field_level_spectrum = compute_spectra.compute_isotropic_power_spectrum_field(sfield)
        array_level_spectrum = _array_compute_spectra.compute_isotropic_power_spectrum_sarray(
            sarray_2d=sarray_2d,
            resolution_2d=resolution_2d,
        )
        numpy.testing.assert_allclose(
            field_level_spectrum.power_spectrum_1d,
            array_level_spectrum.power_spectrum_1d,
        )

    def test_vfield_matches_array_level_kernel(
        self,
    ) -> None:
        num_cells = 8
        resolution_2d = (num_cells, num_cells)
        rng = numpy.random.default_rng(1)
        varray_2d = rng.standard_normal((2, *resolution_2d))
        vfield = _make_vfield(varray_2d)
        field_level_spectrum = compute_spectra.compute_isotropic_power_spectrum_field(vfield)
        array_level_spectrum = _array_compute_spectra.compute_isotropic_power_spectrum_varray(
            varray_2d=varray_2d,
            resolution_2d=resolution_2d,
        )
        numpy.testing.assert_allclose(
            field_level_spectrum.power_spectrum_1d,
            array_level_spectrum.power_spectrum_1d,
        )


class TestFieldTypeValidation(unittest.TestCase):
    """compute_isotropic_power_spectrum_field must reject anything that isn't a
    supported 2D field type, rather than silently misinterpreting it."""

    def test_accepts_sfield(
        self,
    ) -> None:
        sfield = _make_sfield(numpy.ones((8, 8)))
        compute_spectra.compute_isotropic_power_spectrum_field(sfield)

    def test_accepts_vfield(
        self,
    ) -> None:
        vfield = _make_vfield(numpy.ones((2, 8, 8)))
        compute_spectra.compute_isotropic_power_spectrum_field(vfield)

    def test_rejects_bare_array(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            compute_spectra.compute_isotropic_power_spectrum_field(numpy.ones((8, 8)))  # pyright: ignore[reportArgumentType]

    def test_rejects_none(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            compute_spectra.compute_isotropic_power_spectrum_field(None)  # pyright: ignore[reportArgumentType]


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
