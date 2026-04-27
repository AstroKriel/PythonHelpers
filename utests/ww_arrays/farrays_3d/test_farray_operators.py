## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy

## local
from jormi.ww_arrays.farrays_3d import farray_operators

_N_LOW = 8
_N_HIGH = 256
_SSHAPE = (_N_LOW, _N_LOW, _N_LOW)
_VSHAPE = (3, _N_LOW, _N_LOW, _N_LOW)
_CELL_WIDTHS = (1.0 / _N_LOW, 1.0 / _N_LOW, 1.0 / _N_LOW)
_ATOL = 1e-10
_ATOL_FD = 1e-3

##
## === HELPERS
##


def _const_sarray(
    value: float = 1.0,
) -> numpy.ndarray:
    return numpy.full(_SSHAPE, value)


def _const_varray(
    *,
    x0: float = 1.0,
    x1: float = 0.0,
    x2: float = 0.0,
) -> numpy.ndarray:
    varray = numpy.zeros(_VSHAPE)
    varray[0] = x0
    varray[1] = x1
    varray[2] = x2
    return varray


def _cell_centers(
    n: int,
) -> numpy.ndarray:
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
                atol=_ATOL,
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
                atol=_ATOL_FD,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
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
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                expected,
                atol=_ATOL_FD,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
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
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                expected,
                atol=_ATOL_FD,
            ),
        )


class TestDivergence(unittest.TestCase):

    def test_divergence_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_divergence(
            _const_varray(x0=1.0, x1=2.0, x2=3.0),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL,
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
                atol=_ATOL_FD,
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
                atol=_ATOL_FD,
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
                atol=_ATOL_FD,
            ),
        )


class TestCurl(unittest.TestCase):

    def test_curl_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_curl(
            _const_varray(x0=1.0, x1=2.0, x2=3.0),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL,
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
        ## v₁ = sin(2π x₀) → curl[2] = d₀v₁ = 2π cos(2π x₀), others zero
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
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                expected_curl2,
                atol=_ATOL_FD,
            ),
        )

    def test_curl_v2_sin_x1_gives_curl0(
        self,
    ) -> None:
        ## v₂ = sin(2π x₁) → curl[0] = d₁v₂ = 2π cos(2π x₁), others zero
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
                atol=_ATOL_FD,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
            ),
        )

    def test_curl_v0_sin_x2_gives_curl1(
        self,
    ) -> None:
        ## v₀ = sin(2π x₂) → curl[1] = d₂v₀ = 2π cos(2π x₂), others zero
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
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                expected_curl1,
                atol=_ATOL_FD,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
            ),
        )


class TestCrossProduct(unittest.TestCase):

    def test_cross_product_of_vector_with_itself_is_zero(
        self,
    ) -> None:
        varray = _const_varray(x0=1.0, x1=2.0, x2=3.0)
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=varray,
            varray_3d_b=varray,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL,
            ),
        )

    def test_cross_product_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=_const_varray(x0=1.0, x1=0.0, x2=0.0),
            varray_3d_b=_const_varray(x0=0.0, x1=1.0, x2=0.0),
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )

    def test_x0_cross_x1_equals_x2(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=_const_varray(x0=1.0, x1=0.0, x2=0.0),
            varray_3d_b=_const_varray(x0=0.0, x1=1.0, x2=0.0),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                1.0,
                atol=_ATOL,
            ),
        )

    def test_x1_cross_x2_equals_x0(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=_const_varray(x0=0.0, x1=1.0, x2=0.0),
            varray_3d_b=_const_varray(x0=0.0, x1=0.0, x2=1.0),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                1.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
            ),
        )

    def test_x2_cross_x0_equals_x1(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=_const_varray(x0=0.0, x1=0.0, x2=1.0),
            varray_3d_b=_const_varray(x0=1.0, x1=0.0, x2=0.0),
        )
        self.assertTrue(
            numpy.allclose(
                result[0],
                0.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[1],
                1.0,
                atol=_ATOL,
            ),
        )
        self.assertTrue(
            numpy.allclose(
                result[2],
                0.0,
                atol=_ATOL,
            ),
        )

    def test_cross_product_is_anti_commutative(
        self,
    ) -> None:
        varray_a = _const_varray(x0=1.0, x1=2.0, x2=3.0)
        varray_b = _const_varray(x0=4.0, x1=5.0, x2=6.0)
        result_ab = farray_operators.compute_varray_cross_product(
            varray_3d_a=varray_a,
            varray_3d_b=varray_b,
        )
        result_ba = farray_operators.compute_varray_cross_product(
            varray_3d_a=varray_b,
            varray_3d_b=varray_a,
        )
        self.assertTrue(
            numpy.allclose(
                result_ab,
                -result_ba,
                atol=_ATOL,
            ),
        )


class TestDotProduct(unittest.TestCase):

    def test_dot_of_orthogonal_unit_vectors_is_zero(
        self,
    ) -> None:
        result = farray_operators.dot_over_varray_comps(
            varray_3d_a=_const_varray(x0=1.0, x1=0.0, x2=0.0),
            varray_3d_b=_const_varray(x0=0.0, x1=1.0, x2=0.0),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                0.0,
                atol=_ATOL,
            ),
        )

    def test_dot_of_parallel_unit_vectors_is_one(
        self,
    ) -> None:
        varray = _const_varray(x0=1.0, x1=0.0, x2=0.0)
        result = farray_operators.dot_over_varray_comps(
            varray_3d_a=varray,
            varray_3d_b=varray,
        )
        self.assertTrue(
            numpy.allclose(
                result,
                1.0,
                atol=_ATOL,
            ),
        )

    def test_dot_product_of_known_vectors(
        self,
    ) -> None:
        result = farray_operators.dot_over_varray_comps(
            varray_3d_a=_const_varray(x0=1.0, x1=2.0, x2=3.0),
            varray_3d_b=_const_varray(x0=4.0, x1=5.0, x2=6.0),
        )
        self.assertTrue(
            numpy.allclose(
                result,
                32.0,
                atol=_ATOL,
            ),
        )

    def test_dot_product_is_commutative(
        self,
    ) -> None:
        varray_a = _const_varray(x0=1.0, x1=2.0, x2=3.0)
        varray_b = _const_varray(x0=4.0, x1=5.0, x2=6.0)
        result_ab = farray_operators.dot_over_varray_comps(
            varray_3d_a=varray_a,
            varray_3d_b=varray_b,
        )
        result_ba = farray_operators.dot_over_varray_comps(
            varray_3d_a=varray_b,
            varray_3d_b=varray_a,
        )
        self.assertTrue(
            numpy.allclose(
                result_ab,
                result_ba,
                atol=_ATOL,
            ),
        )


class TestSumOfVarrayCompsSquared(unittest.TestCase):

    def test_sum_of_squares_equals_dot_product_with_self(
        self,
    ) -> None:
        varray = _const_varray(x0=1.0, x1=2.0, x2=3.0)
        sum_sq = farray_operators.sum_of_varray_comps_squared(varray_3d=varray)
        dot = farray_operators.dot_over_varray_comps(
            varray_3d_a=varray,
            varray_3d_b=varray,
        )
        self.assertTrue(
            numpy.allclose(
                sum_sq,
                dot,
                atol=_ATOL,
            ),
        )

    def test_constant_vector_sum_of_squares(
        self,
    ) -> None:
        varray = _const_varray(x0=1.0, x1=2.0, x2=3.0)
        result = farray_operators.sum_of_varray_comps_squared(varray_3d=varray)
        self.assertTrue(
            numpy.allclose(
                result,
                14.0,
                atol=_ATOL,
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
                atol=_ATOL,
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
                atol=_ATOL,
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
                atol=_ATOL,
            ),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
