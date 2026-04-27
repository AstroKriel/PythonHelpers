## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_fields.fields_3d import (
    domain_models,
    field_models,
    field_operators,
)

##
## === HELPERS
##

_RESOLUTION = (8, 8, 8)
_DOMAIN_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def _make_3d_udomain(
    *,
    resolution: tuple[int, int, int] = _RESOLUTION,
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = _DOMAIN_BOUNDS,
) -> domain_models.UniformDomain_3D:
    return domain_models.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=domain_bounds,
    )


def _make_constant_sfield(
    *,
    value: float,
    resolution: tuple[int, int, int] = _RESOLUTION,
    label: str = "f",
) -> field_models.ScalarField_3D:
    return field_models.ScalarField_3D.from_3d_sarray(
        sarray_3d=numpy.full(resolution, value),
        udomain_3d=_make_3d_udomain(resolution=resolution),
        field_label=label,
    )


def _make_constant_vfield(
    *,
    value_in_x0: float,
    value_in_x1: float,
    value_in_x2: float,
    resolution: tuple[int, int, int] = _RESOLUTION,
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = _DOMAIN_BOUNDS,
    label: str = "v",
) -> field_models.VectorField_3D:
    varray = numpy.zeros((3, ) + resolution)
    varray[0] = value_in_x0
    varray[1] = value_in_x1
    varray[2] = value_in_x2
    return field_models.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=_make_3d_udomain(
            resolution=resolution,
            domain_bounds=domain_bounds,
        ),
        field_label=label,
    )


##
## === TEST SUITES
##


class TestScalarFieldRms(unittest.TestCase):

    def test_rms_returns_float(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_sfield_rms(
                sfield_3d=_make_constant_sfield(
                    value=2.0,
                ),
            ),
            float,
        )


class TestScalarFieldVolumeIntegral(unittest.TestCase):

    def test_volume_integral_returns_float(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_sfield_volume_integral(
                sfield_3d=_make_constant_sfield(
                    value=1.0,
                ),
            ),
            float,
        )


class TestScalarFieldGradient(unittest.TestCase):

    def test_gradient_returns_vector_field(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_sfield_gradient(
                sfield_3d=_make_constant_sfield(
                    value=1.0,
                ),
            ),
            field_models.VectorField_3D,
        )

    def test_gradient_result_has_same_domain(
        self,
    ):
        sfield = _make_constant_sfield(value=1.0)
        self.assertEqual(
            field_operators.compute_sfield_gradient(sfield_3d=sfield).udomain,
            sfield.udomain,
        )

    def test_gradient_result_has_same_resolution(
        self,
    ):
        sfield = _make_constant_sfield(value=1.0)
        self.assertEqual(
            field_operators.compute_sfield_gradient(sfield_3d=sfield).fdata.sdims_shape,
            sfield.fdata.shape,
        )

    def test_gradient_preserves_sim_time(
        self,
    ):
        sfield = field_models.ScalarField_3D.from_3d_sarray(
            sarray_3d=numpy.ones(_RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="f",
            sim_time=2.5,
        )
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        assert vfield_grad.sim_time is not None
        self.assertAlmostEqual(
            vfield_grad.sim_time,
            2.5,
        )

    def test_gradient_output_buffer_reused_when_compatible(
        self,
    ):
        sfield = _make_constant_sfield(value=1.0)
        array = numpy.empty((3, ) + _RESOLUTION)
        vfield_grad = field_operators.compute_sfield_gradient(
            sfield_3d=sfield,
            varray_3d_out=array,
        )
        self.assertTrue(
            numpy.shares_memory(
                vfield_grad.fdata.farray,
                array,
            ),
        )


class TestVectorFieldMagnitude(unittest.TestCase):

    def test_magnitude_returns_scalar_field(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_vfield_magnitude(
                vfield_3d=_make_constant_vfield(
                    value_in_x0=1.0,
                    value_in_x1=0.0,
                    value_in_x2=0.0,
                ),
            ),
            field_models.ScalarField_3D,
        )

    def test_magnitude_result_has_same_domain(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertEqual(
            field_operators.compute_vfield_magnitude(vfield_3d=vfield).udomain,
            vfield.udomain,
        )


class TestVectorFieldDotProduct(unittest.TestCase):

    def test_dot_product_returns_scalar_field(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertIsInstance(
            field_operators.compute_vfield_dot_product(
                vfield_3d_a=vfield,
                vfield_3d_b=vfield,
            ),
            field_models.ScalarField_3D,
        )

    def test_dot_product_domain_mismatch_raises(
        self,
    ):
        vfield_a = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
            resolution=(4, 4, 4),
        )
        vfield_b = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
        )
        with self.assertRaises(ValueError):
            field_operators.compute_vfield_dot_product(
                vfield_3d_a=vfield_a,
                vfield_3d_b=vfield_b,
            )


class TestVectorFieldCrossProduct(unittest.TestCase):

    def test_cross_product_returns_vector_field(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertIsInstance(
            field_operators.compute_vfield_cross_product(
                vfield_3d_a=vfield,
                vfield_3d_b=vfield,
            ),
            field_models.VectorField_3D,
        )

    def test_cross_product_output_buffer_reused_when_compatible(
        self,
    ):
        vfield_x0 = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        vfield_x1 = _make_constant_vfield(
            value_in_x0=0.0,
            value_in_x1=1.0,
            value_in_x2=0.0,
        )
        array = numpy.empty((3, ) + _RESOLUTION)
        result = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield_x0,
            vfield_3d_b=vfield_x1,
            varray_3d_out=array,
        )
        self.assertTrue(
            numpy.shares_memory(
                result.fdata.farray,
                array,
            ),
        )


class TestVectorFieldDivergence(unittest.TestCase):

    def test_divergence_returns_scalar_field(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_vfield_divergence(
                vfield_3d=_make_constant_vfield(
                    value_in_x0=1.0,
                    value_in_x1=0.0,
                    value_in_x2=0.0,
                ),
            ),
            field_models.ScalarField_3D,
        )

    def test_divergence_result_has_same_domain(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertEqual(
            field_operators.compute_vfield_divergence(vfield_3d=vfield).udomain,
            vfield.udomain,
        )

    def test_divergence_result_has_same_resolution(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertEqual(
            field_operators.compute_vfield_divergence(vfield_3d=vfield).fdata.shape,
            _RESOLUTION,
        )

    def test_divergence_preserves_sim_time(
        self,
    ):
        vfield = field_models.VectorField_3D.from_3d_varray(
            varray_3d=numpy.ones((3, ) + _RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="v",
            sim_time=1.0,
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield_3d=vfield)
        assert sfield_div.sim_time is not None
        self.assertAlmostEqual(
            sfield_div.sim_time,
            1.0,
        )

    def test_divergence_output_buffer_reused_when_compatible(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        array = numpy.empty(_RESOLUTION)
        result = field_operators.compute_vfield_divergence(
            vfield_3d=vfield,
            sarray_3d_out=array,
        )
        self.assertTrue(
            numpy.shares_memory(
                result.fdata.farray,
                array,
            ),
        )


class TestVectorFieldCurl(unittest.TestCase):

    def test_curl_returns_vector_field(
        self,
    ):
        self.assertIsInstance(
            field_operators.compute_vfield_curl(
                vfield_3d=_make_constant_vfield(
                    value_in_x0=1.0,
                    value_in_x1=0.0,
                    value_in_x2=0.0,
                ),
            ),
            field_models.VectorField_3D,
        )

    def test_curl_result_has_same_domain(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertEqual(
            field_operators.compute_vfield_curl(vfield_3d=vfield).udomain,
            vfield.udomain,
        )

    def test_curl_result_has_same_resolution(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertEqual(
            field_operators.compute_vfield_curl(vfield_3d=vfield).fdata.sdims_shape,
            _RESOLUTION,
        )

    def test_curl_preserves_sim_time(
        self,
    ):
        vfield = field_models.VectorField_3D.from_3d_varray(
            varray_3d=numpy.ones((3, ) + _RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="v",
            sim_time=3.0,
        )
        vfield_curl = field_operators.compute_vfield_curl(vfield_3d=vfield)
        assert vfield_curl.sim_time is not None
        self.assertAlmostEqual(
            vfield_curl.sim_time,
            3.0,
        )

    def test_curl_output_buffer_reused_when_compatible(
        self,
    ):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        array = numpy.empty((3, ) + _RESOLUTION)
        result = field_operators.compute_vfield_curl(
            vfield_3d=vfield,
            varray_3d_out=array,
        )
        self.assertTrue(
            numpy.shares_memory(
                result.fdata.farray,
                array,
            ),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
