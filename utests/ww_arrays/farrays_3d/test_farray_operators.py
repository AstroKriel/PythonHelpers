## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d import farray_operators

_N_LOW = 8
_N_HIGH = 256
_SSHAPE = (_N_LOW, _N_LOW, _N_LOW)
_VSHAPE = (3, _N_LOW, _N_LOW, _N_LOW)
_CELL_WIDTHS = (1.0 / _N_LOW, 1.0 / _N_LOW, 1.0 / _N_LOW)
_ATOL_ROUNDOFF = 1e-10
_ATOL_FINITE_DIFF = 1e-3

##
## === HELPERS
##


def _const_sarray(
    value: float = 1.0,
) -> NDArray[Any]:
    return numpy.full(_SSHAPE, value)


def _const_varray(
    *,
    x0: float = 1.0,
    x1: float = 0.0,
    x2: float = 0.0,
) -> NDArray[Any]:
    varray = numpy.zeros(_VSHAPE)
    varray[0] = x0
    varray[1] = x1
    varray[2] = x2
    return varray


def _cell_centers(
    n: int,
) -> NDArray[Any]:
    """Cell centers on [0, 1] for n cells."""
    return (numpy.arange(n) + 0.5) / n


##
## === TEST SUITES
##


class TestScalarArrayRms(unittest.TestCase):

    def test_rms_of_constant_positive(
        self,
    ) -> None:
        self.assertAlmostEqual(
            farray_operators.compute_sarray_rms(
                _const_sarray(
                    3.0,
                ),
            ),
            3.0,
        )

    def test_rms_of_zero_is_zero(
        self,
    ) -> None:
        self.assertAlmostEqual(
            farray_operators.compute_sarray_rms(
                _const_sarray(
                    0.0,
                ),
            ),
            0.0,
        )

    def test_rms_of_alternating_signs_is_one(
        self,
    ) -> None:
        sarray = numpy.ones(_SSHAPE)
        sarray[::2] = -1.0
        self.assertAlmostEqual(
            farray_operators.compute_sarray_rms(sarray),
            1.0,
        )

    def test_rms_returns_float(
        self,
    ) -> None:
        self.assertIsInstance(
            farray_operators.compute_sarray_rms(
                _const_sarray(
                    2.0,
                ),
            ),
            float,
        )


class TestScalarArrayVolumeIntegral(unittest.TestCase):

    def test_integral_of_ones_equals_total_volume(
        self,
    ) -> None:
        cell_volume = (2.0 * 3.0 * 4.0) / _N_LOW**3
        self.assertAlmostEqual(
            farray_operators.compute_sarray_volume_integral(
                numpy.ones(_SSHAPE),
                cell_volume=cell_volume,
            ),
            2.0 * 3.0 * 4.0,
            places=10,
        )

    def test_integral_of_zero_is_zero(
        self,
    ) -> None:
        self.assertAlmostEqual(
            farray_operators.compute_sarray_volume_integral(
                _const_sarray(0.0),
                cell_volume=1.0 / _N_LOW**3,
            ),
            0.0,
        )

    def test_integral_of_constant_equals_value_times_volume(
        self,
    ) -> None:
        self.assertAlmostEqual(
            farray_operators.compute_sarray_volume_integral(
                _const_sarray(5.0),
                cell_volume=1.0 / _N_LOW**3,
            ),
            5.0,
            places=10,
        )

    def test_integral_returns_float(
        self,
    ) -> None:
        self.assertIsInstance(
            farray_operators.compute_sarray_volume_integral(
                _const_sarray(1.0),
                cell_volume=1.0 / _N_LOW**3,
            ),
            float,
        )


class TestGradient(unittest.TestCase):

    def test_gradient_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_sarray_grad(
            _const_sarray(3.0),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_gradient_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_sarray_grad(
            _const_sarray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )

    def test_gradient_sin_x0_along_x0(
        self,
    ) -> None:
        n = _N_HIGH
        x0 = _cell_centers(n)
        sarray = numpy.sin(2.0 * numpy.pi * x0)[:, None, None] * numpy.ones((n, 2, 2))
        result = farray_operators.compute_sarray_grad(
            sarray,
            cell_widths_3d=(1.0 / n, 0.5, 0.5),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0)[:, None, None] * numpy.ones((n, 2, 2)))
        self.assertTrue(
            numpy.allclose(
                result[0],
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_gradient_sin_x1_along_x1(
        self,
    ) -> None:
        n = _N_HIGH
        x1 = _cell_centers(n)
        sarray = numpy.sin(2.0 * numpy.pi * x1)[None, :, None] * numpy.ones((2, n, 2))
        result = farray_operators.compute_sarray_grad(
            sarray,
            cell_widths_3d=(0.5, 1.0 / n, 0.5),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x1)[None, :, None] * numpy.ones((2, n, 2)))
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_gradient_sin_x2_along_x2(
        self,
    ) -> None:
        n = _N_HIGH
        x2 = _cell_centers(n)
        sarray = numpy.sin(2.0 * numpy.pi * x2)[None, None, :] * numpy.ones((2, 2, n))
        result = farray_operators.compute_sarray_grad(
            sarray,
            cell_widths_3d=(0.5, 0.5, 1.0 / n),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x2)[None, None, :] * numpy.ones((2, 2, n)))
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )


class TestDivergence(unittest.TestCase):

    def test_divergence_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_divergence(
            _const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_divergence_output_is_sarray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_divergence(
            _const_varray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _SSHAPE,
        )

    def test_divergence_sin_x0_in_x0(
        self,
    ) -> None:
        n = _N_HIGH
        x0 = _cell_centers(n)
        varray = numpy.zeros((3, n, 2, 2))
        varray[0] = numpy.sin(2.0 * numpy.pi * x0)[:, None, None]
        result = farray_operators.compute_varray_divergence(
            varray,
            cell_widths_3d=(1.0 / n, 0.5, 0.5),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0)[:, None, None] * numpy.ones((n, 2, 2)))
        self.assertTrue(
            numpy.allclose(
                result,
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )

    def test_divergence_sin_x1_in_x1(
        self,
    ) -> None:
        n = _N_HIGH
        x1 = _cell_centers(n)
        varray = numpy.zeros((3, 2, n, 2))
        varray[1] = numpy.sin(2.0 * numpy.pi * x1)[None, :, None]
        result = farray_operators.compute_varray_divergence(
            varray,
            cell_widths_3d=(0.5, 1.0 / n, 0.5),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x1)[None, :, None] * numpy.ones((2, n, 2)))
        self.assertTrue(
            numpy.allclose(
                result,
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )

    def test_divergence_sin_x2_in_x2(
        self,
    ) -> None:
        n = _N_HIGH
        x2 = _cell_centers(n)
        varray = numpy.zeros((3, 2, 2, n))
        varray[2] = numpy.sin(2.0 * numpy.pi * x2)[None, None, :]
        result = farray_operators.compute_varray_divergence(
            varray,
            cell_widths_3d=(0.5, 0.5, 1.0 / n),
        )
        expected = (2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x2)[None, None, :] * numpy.ones((2, 2, n)))
        self.assertTrue(
            numpy.allclose(
                result,
                expected,
                atol=_ATOL_FINITE_DIFF,
            ),
        )


class TestCurl(unittest.TestCase):

    def test_curl_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_curl(
            _const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_curl_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_curl(
            _const_varray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )

    def test_curl_v1_sin_x0_gives_curl2(
        self,
    ) -> None:
        ## v1 = sin(2*pi*x0) -> curl[2] = d0(v1) = 2*pi*cos(2*pi*x0), others zero
        n = _N_HIGH
        x0 = _cell_centers(n)
        varray = numpy.zeros((3, n, 2, 2))
        varray[1] = numpy.sin(2.0 * numpy.pi * x0)[:, None, None]
        result = farray_operators.compute_varray_curl(
            varray,
            cell_widths_3d=(1.0 / n, 0.5, 0.5),
        )
        expected_curl2 = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x0)[:, None, None] * numpy.ones((n, 2, 2))
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                expected_curl2,
                atol=_ATOL_FINITE_DIFF,
            ),
        )

    def test_curl_v2_sin_x1_gives_curl0(
        self,
    ) -> None:
        ## v2 = sin(2*pi*x1) -> curl[0] = d1(v2) = 2*pi*cos(2*pi*x1), others zero
        n = _N_HIGH
        x1 = _cell_centers(n)
        varray = numpy.zeros((3, 2, n, 2))
        varray[2] = numpy.sin(2.0 * numpy.pi * x1)[None, :, None]
        result = farray_operators.compute_varray_curl(
            varray,
            cell_widths_3d=(0.5, 1.0 / n, 0.5),
        )
        expected_curl0 = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x1)[None, :, None] * numpy.ones((2, n, 2))
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                expected_curl0,
                atol=_ATOL_FINITE_DIFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_curl_v0_sin_x2_gives_curl1(
        self,
    ) -> None:
        ## v0 = sin(2*pi*x2) -> curl[1] = d2(v0) = 2*pi*cos(2*pi*x2), others zero
        n = _N_HIGH
        x2 = _cell_centers(n)
        varray = numpy.zeros((3, 2, 2, n))
        varray[0] = numpy.sin(2.0 * numpy.pi * x2)[None, None, :]
        result = farray_operators.compute_varray_curl(
            varray,
            cell_widths_3d=(0.5, 0.5, 1.0 / n),
        )
        expected_curl1 = (
            2.0 * numpy.pi * numpy.cos(2.0 * numpy.pi * x2)[None, None, :] * numpy.ones((2, 2, n))
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                expected_curl1,
                atol=_ATOL_FINITE_DIFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )


class TestCrossProduct(unittest.TestCase):

    def test_cross_product_of_vector_with_itself_is_zero(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        result = farray_operators.compute_varray_cross_product(
            f_varray_3d=varray,
            g_varray_3d=varray,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_cross_product_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            f_varray_3d=_const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
            g_varray_3d=_const_varray(
                x0=0.0,
                x1=1.0,
                x2=0.0,
            ),
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )

    def test_x0_cross_x1_equals_x2(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            f_varray_3d=_const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
            g_varray_3d=_const_varray(
                x0=0.0,
                x1=1.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                1.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_x1_cross_x2_equals_x0(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            f_varray_3d=_const_varray(
                x0=0.0,
                x1=1.0,
                x2=0.0,
            ),
            g_varray_3d=_const_varray(
                x0=0.0,
                x1=0.0,
                x2=1.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                1.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_x2_cross_x0_equals_x1(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            f_varray_3d=_const_varray(
                x0=0.0,
                x1=0.0,
                x2=1.0,
            ),
            g_varray_3d=_const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                1.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_cross_product_is_anti_commutative(
        self,
    ) -> None:
        varray_a = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        varray_b = _const_varray(
            x0=4.0,
            x1=5.0,
            x2=6.0,
        )
        result_ab = farray_operators.compute_varray_cross_product(
            f_varray_3d=varray_a,
            g_varray_3d=varray_b,
        )
        result_ba = farray_operators.compute_varray_cross_product(
            f_varray_3d=varray_b,
            g_varray_3d=varray_a,
        )
        self.assertTrue(
            numpy.allclose(
                result_ab,
                -result_ba,
                atol=_ATOL_ROUNDOFF,
            ),
        )


class TestDotProduct(unittest.TestCase):

    def test_dot_of_orthogonal_unit_vectors_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=_const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
            g_varray_3d=_const_varray(
                x0=0.0,
                x1=1.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_dot_of_parallel_unit_vectors_is_one(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=0.0,
            x2=0.0,
        )
        result = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=varray,
            g_varray_3d=varray,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                1.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_dot_product_of_known_vectors(
        self,
    ) -> None:
        result = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=_const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            g_varray_3d=_const_varray(
                x0=4.0,
                x1=5.0,
                x2=6.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                32.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_dot_product_is_commutative(
        self,
    ) -> None:
        varray_a = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        varray_b = _const_varray(
            x0=4.0,
            x1=5.0,
            x2=6.0,
        )
        result_ab = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=varray_a,
            g_varray_3d=varray_b,
        )
        result_ba = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=varray_b,
            g_varray_3d=varray_a,
        )
        self.assertTrue(
            numpy.allclose(
                result_ab,
                result_ba,
                atol=_ATOL_ROUNDOFF,
            ),
        )


class TestSumOfVarrayCompsSquared(unittest.TestCase):

    def test_sum_of_squares_equals_dot_product_with_self(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        sum_sq = farray_operators.compute_sum_of_varray_comps_squared(varray_3d=varray)
        dot = farray_operators.compute_dot_over_varray_comps(
            f_varray_3d=varray,
            g_varray_3d=varray,
        )
        self.assertTrue(
            numpy.allclose(
                sum_sq,
                dot,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_constant_vector_sum_of_squares(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        result = farray_operators.compute_sum_of_varray_comps_squared(varray_3d=varray)
        self.assertTrue(
            numpy.allclose(
                result,
                14.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )


class TestMagnitude(unittest.TestCase):

    def test_magnitude_is_non_negative(
        self,
    ) -> None:
        result = farray_operators.compute_varray_magnitude(
            _const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
        )
        self.assertTrue(
            numpy.all(
                result >= 0.0,
            ),
        )

    def test_magnitude_of_known_vector(
        self,
    ) -> None:
        result = farray_operators.compute_varray_magnitude(
            _const_varray(
                x0=3.0,
                x1=4.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                5.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_magnitude_of_zero_vector_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_magnitude(
            _const_varray(
                x0=0.0,
                x1=0.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_magnitude_of_unit_vector_is_one(
        self,
    ) -> None:
        result = farray_operators.compute_varray_magnitude(
            _const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                1.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )


class TestNormalized(unittest.TestCase):

    def test_normalized_has_unit_magnitude(
        self,
    ) -> None:
        rng = numpy.random.default_rng(0)
        varray = rng.standard_normal(_VSHAPE) + 1.0  ## keep well away from zero
        result = farray_operators.compute_varray_normalized(varray)
        magnitude = farray_operators.compute_varray_magnitude(result)
        numpy.testing.assert_allclose(magnitude, 1.0, atol=1e-8)

    def test_normalized_preserves_direction(
        self,
    ) -> None:
        varray = _const_varray(x0=3.0, x1=4.0, x2=0.0)
        result = farray_operators.compute_varray_normalized(varray)
        numpy.testing.assert_allclose(result[0], 0.6, atol=_ATOL_ROUNDOFF)
        numpy.testing.assert_allclose(result[1], 0.8, atol=_ATOL_ROUNDOFF)
        numpy.testing.assert_allclose(result[2], 0.0, atol=_ATOL_ROUNDOFF)

    def test_zero_vector_is_zeroed_not_nan(
        self,
    ) -> None:
        varray = _const_varray(x0=0.0, x1=0.0, x2=0.0)
        result = farray_operators.compute_varray_normalized(varray)
        self.assertTrue(numpy.all(numpy.isfinite(result)))
        numpy.testing.assert_allclose(result, 0.0, atol=_ATOL_ROUNDOFF)


class TestR2TarrayDoubleDot(unittest.TestCase):

    def test_axis_aligned_vector_picks_out_diagonal_entry(
        self,
    ) -> None:
        ## v = (1, 0, 0), T = diag(a, b, c) -> v_i v_j T_ij = a
        r2tarray = numpy.zeros((3, 3, *_SSHAPE))
        r2tarray[0, 0] = 5.0
        r2tarray[1, 1] = -2.0
        r2tarray[2, 2] = 7.0
        varray = _const_varray(x0=1.0, x1=0.0, x2=0.0)
        result = farray_operators.compute_varray_r2tarray_double_dot(
            varray_3d=varray,
            r2tarray_3d=r2tarray,
        )
        numpy.testing.assert_allclose(result, 5.0, atol=_ATOL_ROUNDOFF)

    def test_matches_hand_computed_quadratic_form(
        self,
    ) -> None:
        ## v = (1, 1, 0), T = diag(a, b, c) -> v_i v_j T_ij = a + b
        r2tarray = numpy.zeros((3, 3, *_SSHAPE))
        r2tarray[0, 0] = 5.0
        r2tarray[1, 1] = -2.0
        r2tarray[2, 2] = 7.0
        varray = _const_varray(x0=1.0, x1=1.0, x2=0.0)
        result = farray_operators.compute_varray_r2tarray_double_dot(
            varray_3d=varray,
            r2tarray_3d=r2tarray,
        )
        numpy.testing.assert_allclose(result, 3.0, atol=_ATOL_ROUNDOFF)

    def test_rejects_mismatched_domains(
        self,
    ) -> None:
        r2tarray = numpy.zeros((3, 3, 4, 4, 4))
        varray = _const_varray()
        with self.assertRaises(ValueError):
            farray_operators.compute_varray_r2tarray_double_dot(
                varray_3d=varray,
                r2tarray_3d=r2tarray,
            )


class TestR2TarrayMagnitude(unittest.TestCase):

    def test_magnitude_is_non_negative(
        self,
    ) -> None:
        r2tarray = numpy.zeros((3, 3, *_SSHAPE))
        r2tarray[0, 0] = -5.0
        r2tarray[1, 2] = 3.0
        result = farray_operators.compute_r2tarray_magnitude(r2tarray)
        self.assertTrue(numpy.all(result >= 0.0))

    def test_magnitude_of_known_tensor(
        self,
    ) -> None:
        ## diag(3, 4, 0) -> sqrt(3^2 + 4^2) = 5
        r2tarray = numpy.zeros((3, 3, *_SSHAPE))
        r2tarray[0, 0] = 3.0
        r2tarray[1, 1] = 4.0
        result = farray_operators.compute_r2tarray_magnitude(r2tarray)
        numpy.testing.assert_allclose(result, 5.0, atol=_ATOL_ROUNDOFF)

    def test_magnitude_of_zero_tensor_is_zero(
        self,
    ) -> None:
        r2tarray = numpy.zeros((3, 3, *_SSHAPE))
        result = farray_operators.compute_r2tarray_magnitude(r2tarray)
        numpy.testing.assert_allclose(result, 0.0, atol=_ATOL_ROUNDOFF)


class TestStrainRate(unittest.TestCase):

    def test_constant_field_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_strain_rate(
            _const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL_ROUNDOFF,
            ),
        )

    def test_output_shape(
        self,
    ) -> None:
        result = farray_operators.compute_varray_strain_rate(
            _const_varray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            (3, 3, *_SSHAPE),
        )

    def test_symmetric_for_any_field(
        self,
    ) -> None:
        ## the exact bug this guards against: 0.5 * A + A^T is not symmetric unless A
        ## already is; 0.5 * (A + A^T) is symmetric for any A. A field with genuinely
        ## asymmetric gradient (nonzero curl, e.g. this random one) is required to
        ## actually exercise the difference between the two.
        rng = numpy.random.default_rng(0)
        varray = rng.standard_normal(_VSHAPE)
        result = farray_operators.compute_varray_strain_rate(
            varray,
            cell_widths_3d=_CELL_WIDTHS,
        )
        numpy.testing.assert_allclose(
            result,
            numpy.transpose(result, axes=(1, 0, 2, 3, 4)),
            atol=_ATOL_ROUNDOFF,
        )

    def test_traceless_for_any_field(
        self,
    ) -> None:
        rng = numpy.random.default_rng(1)
        varray = rng.standard_normal(_VSHAPE)
        result = farray_operators.compute_varray_strain_rate(
            varray,
            cell_widths_3d=_CELL_WIDTHS,
        )
        trace = numpy.trace(result, axis1=0, axis2=1)
        numpy.testing.assert_allclose(
            trace,
            0.0,
            atol=_ATOL_ROUNDOFF,
        )

    def test_matches_analytic_off_diagonal_strain(
        self,
    ) -> None:
        ## v0 = sin(2*pi*x1): grad[0,1] = 2*pi*cos(2*pi*x1), every other grad entry is
        ## exactly zero (an asymmetric gradient), so S_01 = S_10 = 0.5 * grad[0,1] and
        ## every other entry is exactly zero
        n = _N_HIGH
        x1 = _cell_centers(n)
        varray = numpy.zeros((3, 2, n, 2))
        varray[0] = numpy.sin(2.0 * numpy.pi * x1)[None, :, None]
        result = farray_operators.compute_varray_strain_rate(
            varray,
            cell_widths_3d=(0.5, 1.0 / n, 0.5),
        )
        expected_s01 = numpy.pi * numpy.cos(2.0 * numpy.pi * x1)[None, :, None] * numpy.ones((2, n, 2))
        numpy.testing.assert_allclose(result[0, 1], expected_s01, atol=_ATOL_FINITE_DIFF)
        numpy.testing.assert_allclose(result[1, 0], expected_s01, atol=_ATOL_FINITE_DIFF)
        for row_index in range(3):
            for col_index in range(3):
                if {row_index, col_index} == {0, 1}:
                    continue
                numpy.testing.assert_allclose(
                    result[row_index, col_index],
                    0.0,
                    atol=_ATOL_FINITE_DIFF,
                    err_msg=f"unexpected nonzero S[{row_index},{col_index}]",
                )

    def test_kinetic_dissipation_matches_divergence_of_strain_rate(
        self,
    ) -> None:
        ## regression guard for the refactor: kinetic dissipation must equal the
        ## divergence of the exact strain-rate tensor this module exposes directly
        rng = numpy.random.default_rng(2)
        varray = rng.standard_normal(_VSHAPE)
        dissipation = farray_operators.compute_varray_kinetic_dissipation(
            varray,
            cell_widths_3d=_CELL_WIDTHS,
        )
        strain_rate = farray_operators.compute_varray_strain_rate(
            varray,
            cell_widths_3d=_CELL_WIDTHS,
        )
        expected = farray_operators.compute_r2tarray_divergence(
            r2tarray_3d=strain_rate,
            cell_widths_3d=_CELL_WIDTHS,
        )
        numpy.testing.assert_allclose(dissipation, expected, atol=_ATOL_ROUNDOFF)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
