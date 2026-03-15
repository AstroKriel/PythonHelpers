## { TEST

##
## === DEPENDENCIES
##

import numpy
import unittest

from jormi.ww_fields.fields_3d import (
    domain_types,
    field_types,
    field_operators,
)

##
## === HELPERS
##

_RESOLUTION = (8, 8, 8)
_DOMAIN_BOUNDS = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def _make_3d_udomain(
    resolution: tuple = _RESOLUTION,
    domain_bounds: tuple = _DOMAIN_BOUNDS,
) -> domain_types.UniformDomain_3D:
    return domain_types.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=resolution,
        domain_bounds=domain_bounds,
    )


def _make_constant_sfield(
    value: float,
    resolution: tuple = _RESOLUTION,
    label: str = "f",
) -> field_types.ScalarField_3D:
    sarray = numpy.full(resolution, value)
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray,
        udomain_3d=_make_3d_udomain(resolution),
        field_label=label,
    )


def _make_constant_vfield(
    value_in_x0: float,
    value_in_x1: float,
    value_in_x2: float,
    resolution: tuple = _RESOLUTION,
    label: str = "v",
) -> field_types.VectorField_3D:
    varray = numpy.zeros((3, ) + resolution)
    varray[0] = value_in_x0
    varray[1] = value_in_x1
    varray[2] = value_in_x2
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=_make_3d_udomain(resolution),
        field_label=label,
    )


##
## === TEST SUITES
##


class TestScalarFieldRms(unittest.TestCase):

    def test_rms_of_constant_positive_field(self):
        sfield = _make_constant_sfield(value=3.0)
        self.assertAlmostEqual(
            field_operators.compute_sfield_rms(sfield),
            3.0,
        )

    def test_rms_of_zero_field(self):
        sfield = _make_constant_sfield(value=0.0)
        self.assertAlmostEqual(
            field_operators.compute_sfield_rms(sfield),
            0.0,
        )

    def test_rms_of_alternating_signs_is_positive(self):
        sarray = numpy.ones(_RESOLUTION)
        sarray[::2] = -1.0
        sfield = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray,
            udomain_3d=_make_3d_udomain(),
            field_label="f",
        )
        self.assertAlmostEqual(
            field_operators.compute_sfield_rms(sfield),
            1.0,
        )

    def test_rms_returns_float(self):
        sfield = _make_constant_sfield(value=2.0)
        self.assertIsInstance(
            field_operators.compute_sfield_rms(sfield),
            float,
        )


class TestScalarFieldVolumeIntegral(unittest.TestCase):

    def test_integral_of_ones_equals_total_volume(self):
        domain = _make_3d_udomain(domain_bounds=((0.0, 2.0), (0.0, 3.0), (0.0, 4.0)))
        sfield = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=numpy.ones(_RESOLUTION),
            udomain_3d=domain,
            field_label="ones",
        )
        self.assertAlmostEqual(
            field_operators.compute_sfield_volume_integral(sfield),
            2.0 * 3.0 * 4.0,
            places=10,
        )

    def test_integral_of_zero_field_is_zero(self):
        sfield = _make_constant_sfield(value=0.0)
        self.assertAlmostEqual(
            field_operators.compute_sfield_volume_integral(sfield),
            0.0,
        )

    def test_integral_of_constant_equals_value_times_volume(self):
        value = 5.0
        domain = _make_3d_udomain()
        sfield = _make_constant_sfield(value=value)
        self.assertAlmostEqual(
            field_operators.compute_sfield_volume_integral(sfield),
            value * domain.total_volume,
            places=10,
        )

    def test_integral_returns_float(self):
        sfield = _make_constant_sfield(value=1.0)
        self.assertIsInstance(
            field_operators.compute_sfield_volume_integral(sfield),
            float,
        )


class TestScalarFieldGradient(unittest.TestCase):

    def test_gradient_of_constant_field_is_zero(self):
        sfield = _make_constant_sfield(value=7.0)
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_varray(vfield_grad),
                0.0,
            ),
        )

    def test_gradient_returns_vector_field(self):
        sfield = _make_constant_sfield(value=1.0)
        self.assertIsInstance(
            field_operators.compute_sfield_gradient(sfield_3d=sfield),
            field_types.VectorField_3D,
        )

    def test_gradient_result_has_same_domain(self):
        sfield = _make_constant_sfield(value=1.0)
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        self.assertEqual(
            vfield_grad.udomain,
            sfield.udomain,
        )

    def test_gradient_result_has_same_resolution(self):
        sfield = _make_constant_sfield(value=1.0)
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        self.assertEqual(
            vfield_grad.fdata.sdims_shape,
            sfield.fdata.shape,
        )

    def test_gradient_preserves_sim_time(self):
        sfield = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=numpy.ones(_RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="f",
            sim_time=2.5,
        )
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        self.assertAlmostEqual(
            vfield_grad.sim_time,
            2.5,  # type: ignore[arg-type]
        )

    def test_gradient_sinusoidal_x0_component(self):
        N = 256
        resolution = (N, 2, 2)
        domain = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=resolution,
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        x0_centers, _, _ = domain.cell_centers
        sarray = numpy.sin(2.0 * numpy.pi * x0_centers)[:, None, None] * numpy.ones(resolution)
        sfield = field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray,
            udomain_3d=domain,
            field_label="sin_x0",
        )
        vfield_grad = field_operators.compute_sfield_gradient(sfield_3d=sfield)
        varray = field_types.extract_3d_varray(vfield_grad)
        expected_dx0 = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0_centers)[:, None, None] * numpy.ones(resolution)
        )
        self.assertTrue(
            numpy.allclose(
                varray[0],
                expected_dx0,
                atol=1e-3,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray[1],
                0.0,
                atol=1e-10,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray[2],
                0.0,
                atol=1e-10,
            ),
        )

    def test_gradient_output_buffer_reused_when_compatible(self):
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

    def test_magnitude_of_known_vector(self):
        vfield = _make_constant_vfield(
            value_in_x0=3.0,
            value_in_x1=4.0,
            value_in_x2=0.0,
        )
        sfield_magn = field_operators.compute_vfield_magnitude(vfield)
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_magn),
                5.0,
            ),
        )

    def test_magnitude_of_zero_vector_is_zero(self):
        vfield = _make_constant_vfield(
            value_in_x0=0.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_magn = field_operators.compute_vfield_magnitude(vfield)
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_magn),
                0.0,
            ),
        )

    def test_magnitude_of_unit_vector_is_one(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_magn = field_operators.compute_vfield_magnitude(vfield)
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_magn),
                1.0,
            ),
        )

    def test_magnitude_returns_scalar_field(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertIsInstance(
            field_operators.compute_vfield_magnitude(vfield),
            field_types.ScalarField_3D,
        )

    def test_magnitude_result_has_same_domain(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_magn = field_operators.compute_vfield_magnitude(vfield)
        self.assertEqual(
            sfield_magn.udomain,
            vfield.udomain,
        )


class TestVectorFieldDotProduct(unittest.TestCase):

    def test_dot_of_orthogonal_unit_vectors_is_zero(self):
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
        sfield_dot = field_operators.compute_vfield_dot_product(
            vfield_3d_a=vfield_x0,
            vfield_3d_b=vfield_x1,
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_dot),
                0.0,
            ),
        )

    def test_dot_of_parallel_unit_vectors_is_one(self):
        vfield_x0 = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_dot = field_operators.compute_vfield_dot_product(
            vfield_3d_a=vfield_x0,
            vfield_3d_b=vfield_x0,
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_dot),
                1.0,
            ),
        )

    def test_dot_product_of_known_vectors(self):
        vfield_a = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        vfield_b = _make_constant_vfield(
            value_in_x0=4.0,
            value_in_x1=5.0,
            value_in_x2=6.0,
        )
        sfield_dot = field_operators.compute_vfield_dot_product(
            vfield_3d_a=vfield_a,
            vfield_3d_b=vfield_b,
        )
        self.assertTrue(numpy.allclose(field_types.extract_3d_sarray(sfield_dot), 32.0))

    def test_dot_product_is_commutative(self):
        vfield_a = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        vfield_b = _make_constant_vfield(
            value_in_x0=4.0,
            value_in_x1=5.0,
            value_in_x2=6.0,
        )
        sfield_ab = field_operators.compute_vfield_dot_product(
            vfield_3d_a=vfield_a,
            vfield_3d_b=vfield_b,
        )
        sfield_ba = field_operators.compute_vfield_dot_product(
            vfield_3d_a=vfield_b,
            vfield_3d_b=vfield_a,
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_ab),
                field_types.extract_3d_sarray(sfield_ba),
            ),
        )

    def test_dot_product_returns_scalar_field(self):
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
            field_types.ScalarField_3D,
        )

    def test_dot_product_domain_mismatch_raises(self):
        vfield_a = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
            resolution=(4, 4, 4),
        )
        different_domain = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=(4, 4, 4),
            domain_bounds=((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
        )
        vfield_b = field_types.VectorField_3D.from_3d_varray(
            varray_3d=numpy.ones((3, 4, 4, 4)),
            udomain_3d=different_domain,
            field_label="v_diff",
        )
        with self.assertRaises(ValueError):
            field_operators.compute_vfield_dot_product(
                vfield_3d_a=vfield_a,
                vfield_3d_b=vfield_b,
            )


class TestVectorFieldCrossProduct(unittest.TestCase):

    def test_x0_cross_x1_equals_x2(self):
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
        vfield_cross = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield_x0,
            vfield_3d_b=vfield_x1,
        )
        varray = field_types.extract_3d_varray(vfield_cross)
        self.assertTrue(
            numpy.allclose(
                varray[0],
                0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray[1],
                0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray[2],
                1.0,
            ),
        )

    def test_x1_cross_x0_equals_minus_x2(self):
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
        vfield_cross = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield_x1,
            vfield_3d_b=vfield_x0,
        )
        varray = field_types.extract_3d_varray(vfield_cross)
        self.assertTrue(numpy.allclose(varray[2], -1.0))

    def test_cross_product_anti_commutative(self):
        vfield_a = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        vfield_b = _make_constant_vfield(
            value_in_x0=4.0,
            value_in_x1=5.0,
            value_in_x2=6.0,
        )
        vfield_ab = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield_a,
            vfield_3d_b=vfield_b,
        )
        vfield_ba = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield_b,
            vfield_3d_b=vfield_a,
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_varray(vfield_ab),
                -field_types.extract_3d_varray(vfield_ba),
            ),
        )

    def test_self_cross_product_is_zero(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        vfield_cross = field_operators.compute_vfield_cross_product(
            vfield_3d_a=vfield,
            vfield_3d_b=vfield,
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_varray(vfield_cross),
                0.0,
            ),
        )

    def test_cross_product_returns_vector_field(self):
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
            field_types.VectorField_3D,
        )

    def test_output_buffer_reused_when_compatible(self):
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
        self.assertTrue(numpy.shares_memory(result.fdata.farray, array))


class TestVectorFieldDivergence(unittest.TestCase):

    def test_divergence_of_constant_field_is_zero(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield)
        self.assertTrue(numpy.allclose(field_types.extract_3d_sarray(sfield_div), 0.0))

    def test_divergence_returns_scalar_field(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertIsInstance(
            field_operators.compute_vfield_divergence(vfield),
            field_types.ScalarField_3D,
        )

    def test_divergence_result_has_same_domain(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield)
        self.assertEqual(
            sfield_div.udomain,
            vfield.udomain,
        )

    def test_divergence_result_has_same_resolution(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield)
        self.assertEqual(
            sfield_div.fdata.shape,
            _RESOLUTION,
        )

    def test_divergence_preserves_sim_time(self):
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=numpy.ones((3, ) + _RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="v",
            sim_time=1.0,
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield)
        self.assertAlmostEqual(
            sfield_div.sim_time,
            1.0,  # type: ignore[arg-type]
        )

    def test_divergence_sinusoidal_field(self):
        N = 256
        resolution = (N, 2, 2)
        domain = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=resolution,
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        x0_centers, _, _ = domain.cell_centers
        varray = numpy.zeros((3, ) + resolution)
        varray[0] = numpy.sin(2.0 * numpy.pi * x0_centers)[:, None, None]
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=domain,
            field_label="v_sin",
        )
        sfield_div = field_operators.compute_vfield_divergence(vfield)
        expected = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0_centers)[:, None, None] * numpy.ones(resolution)
        )
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_sarray(sfield_div),
                expected,
                atol=1e-3,
            ),
        )

    def test_output_buffer_reused_when_compatible(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        array = numpy.empty(_RESOLUTION)
        result = field_operators.compute_vfield_divergence(
            vfield,
            sarray_3d_out=array,
        )
        self.assertTrue(
            numpy.shares_memory(
                result.fdata.farray,
                array,
            ),
        )


class TestVectorFieldCurl(unittest.TestCase):

    def test_curl_of_constant_field_is_zero(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=2.0,
            value_in_x2=3.0,
        )
        vfield_curl = field_operators.compute_vfield_curl(vfield)
        self.assertTrue(
            numpy.allclose(
                field_types.extract_3d_varray(vfield_curl),
                0.0,
            ),
        )

    def test_curl_returns_vector_field(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        self.assertIsInstance(
            field_operators.compute_vfield_curl(vfield),
            field_types.VectorField_3D,
        )

    def test_curl_result_has_same_domain(self):
        vfield = _make_constant_vfield(value_in_x0=1.0, value_in_x1=0.0, value_in_x2=0.0)
        vfield_curl = field_operators.compute_vfield_curl(vfield)
        self.assertEqual(
            vfield_curl.udomain,
            vfield.udomain,
        )

    def test_curl_result_has_same_resolution(self):
        vfield = _make_constant_vfield(value_in_x0=1.0, value_in_x1=0.0, value_in_x2=0.0)
        vfield_curl = field_operators.compute_vfield_curl(vfield)
        self.assertEqual(
            vfield_curl.fdata.sdims_shape,
            _RESOLUTION,
        )

    def test_curl_preserves_sim_time(self):
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=numpy.ones((3, ) + _RESOLUTION),
            udomain_3d=_make_3d_udomain(),
            field_label="v",
            sim_time=3.0,
        )
        vfield_curl = field_operators.compute_vfield_curl(vfield)
        self.assertAlmostEqual(
            vfield_curl.sim_time,
            3.0,  # type: ignore[arg-type]
        )

    def test_curl_sinusoidal_field(self):
        N = 256
        resolution = (N, 2, 2)
        domain = domain_types.UniformDomain_3D(
            periodicity=(True, True, True),
            resolution=resolution,
            domain_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        x0_centers, _, _ = domain.cell_centers
        varray = numpy.zeros((3, ) + resolution)
        varray[1] = numpy.sin(2.0 * numpy.pi * x0_centers)[:, None, None]
        vfield = field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray,
            udomain_3d=domain,
            field_label="v_sin",
        )
        vfield_curl = field_operators.compute_vfield_curl(vfield)
        varray_curl = field_types.extract_3d_varray(vfield_curl)
        expected_x2 = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0_centers)[:, None, None] * numpy.ones(resolution)
        )
        self.assertTrue(
            numpy.allclose(
                varray_curl[0],
                0.0,
                atol=1e-10,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray_curl[1],
                0.0,
                atol=1e-10,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                varray_curl[2],
                expected_x2,
                atol=1e-3,
            ),
        )

    def test_output_buffer_reused_when_compatible(self):
        vfield = _make_constant_vfield(
            value_in_x0=1.0,
            value_in_x1=0.0,
            value_in_x2=0.0,
        )
        array = numpy.empty((3, ) + _RESOLUTION)
        result = field_operators.compute_vfield_curl(
            vfield,
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

## } TEST
