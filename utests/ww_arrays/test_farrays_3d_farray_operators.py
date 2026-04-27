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

_N = 8
_SSHAPE = (_N, _N, _N)
_VSHAPE = (3, _N, _N, _N)
_CELL_WIDTHS = (1.0 / _N, 1.0 / _N, 1.0 / _N)
_ATOL = 1e-10

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


##
## === TEST SUITES
##


class TestGradient(unittest.TestCase):

    def test_gradient_of_constant_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_sarray_grad(
            sarray_3d=_const_sarray(3.0),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(result, 0.0, atol=_ATOL),
        )

    def test_gradient_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_sarray_grad(
            sarray_3d=_const_sarray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )


class TestCurl(unittest.TestCase):

    def test_curl_of_constant_vector_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_curl(
            varray_3d=_const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(result, 0.0, atol=_ATOL),
        )

    def test_curl_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_curl(
            varray_3d=_const_varray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
        )


class TestDivergence(unittest.TestCase):

    def test_divergence_of_constant_vector_is_zero(
        self,
    ) -> None:
        result = farray_operators.compute_varray_divergence(
            varray_3d=_const_varray(
                x0=1.0,
                x1=2.0,
                x2=3.0,
            ),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertTrue(
            numpy.allclose(result, 0.0, atol=_ATOL),
        )

    def test_divergence_output_is_sarray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_divergence(
            varray_3d=_const_varray(),
            cell_widths_3d=_CELL_WIDTHS,
        )
        self.assertEqual(
            result.shape,
            _SSHAPE,
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
            varray_3d_a=varray,
            varray_3d_b=varray,
        )
        self.assertTrue(
            numpy.allclose(result, 0.0, atol=_ATOL),
        )

    def test_cross_product_output_is_varray(
        self,
    ) -> None:
        result = farray_operators.compute_varray_cross_product(
            varray_3d_a=_const_varray(
                x0=1.0,
                x1=0.0,
                x2=0.0,
            ),
            varray_3d_b=_const_varray(
                x0=0.0,
                x1=1.0,
                x2=0.0,
            ),
        )
        self.assertEqual(
            result.shape,
            _VSHAPE,
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
        sum_sq = farray_operators.sum_of_varray_comps_squared(varray_3d=varray)
        dot = farray_operators.dot_over_varray_comps(
            varray_3d_a=varray,
            varray_3d_b=varray,
        )
        self.assertTrue(
            numpy.allclose(sum_sq, dot, atol=_ATOL),
        )

    def test_constant_vector_sum_of_squares(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        result = farray_operators.sum_of_varray_comps_squared(varray_3d=varray)
        self.assertTrue(
            numpy.allclose(result, 14.0, atol=_ATOL),
        )


class TestMagnitude(unittest.TestCase):

    def test_magnitude_is_non_negative(
        self,
    ) -> None:
        varray = _const_varray(
            x0=1.0,
            x1=2.0,
            x2=3.0,
        )
        result = farray_operators.compute_varray_magnitude(varray_3d=varray)
        self.assertTrue(
            numpy.all(result >= 0.0),
        )

    def test_magnitude_of_constant_vector(
        self,
    ) -> None:
        varray = _const_varray(
            x0=3.0,
            x1=4.0,
            x2=0.0,
        )
        result = farray_operators.compute_varray_magnitude(varray_3d=varray)
        self.assertTrue(
            numpy.allclose(result, 5.0, atol=_ATOL),
        )


## } U-TEST
