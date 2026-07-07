## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_arrays.farrays_3d_unstructured import gradient_operators

##
## === HELPERS
##


def _generate_scattered_positions(
    *,
    num_points: int,
    rng: numpy.random.Generator,
    domain_size: float = 1.0,
) -> NDArray[numpy.float64]:
    return rng.uniform(
        0.0,
        domain_size,
        size=(num_points, 3),
    )


def _generate_clustered_positions(
    *,
    num_points_dense: int,
    num_points_sparse: int,
    rng: numpy.random.Generator,
) -> NDArray[numpy.float64]:
    """
    Two length scales in one point cloud: a tight cluster plus a sparse halo.

    Mirrors the orders-of-magnitude cell-size contrast between the dense Arepo
    disc and the diffuse CGM that motivated `compute_gradient_wls` in the first place.
    """
    dense_positions = rng.uniform(
        0.0,
        0.01,
        size=(num_points_dense, 3),
    )
    sparse_positions = rng.uniform(
        -5.0,
        5.0,
        size=(num_points_sparse, 3),
    )
    return numpy.concatenate(
        [dense_positions, sparse_positions],
        axis=0,
    )


def _evaluate_linear_sarray(
    *,
    positions: NDArray[numpy.float64],
    coeffs: NDArray[numpy.float64],
    offset: float,
) -> NDArray[numpy.float64]:
    """f(x) = coeffs . x + offset; exact gradient is `coeffs` everywhere."""
    return positions @ coeffs + offset


def _evaluate_linear_varray(
    *,
    positions: NDArray[numpy.float64],
    coeff_matrix: NDArray[numpy.float64],
    offset: NDArray[numpy.float64],
) -> NDArray[numpy.float64]:
    """f_m(x) = coeff_matrix[m] . x + offset[m]; exact gradient is `coeff_matrix` everywhere."""
    return positions @ coeff_matrix.T + offset


##
## === TEST SUITES
##


class TestGradientWLS_LinearScalarField(unittest.TestCase):
    """
    WLS fits a local linear model, so it is exact for a genuinely linear field,
    regardless of point layout. This isolates implementation bugs (sign errors,
    transposed axes, wrong weighting) from ordinary approximation error.
    """

    def test_recovers_constant_gradient_on_scattered_points(
        self,
    ):
        rng = numpy.random.default_rng(seed=0)
        positions = _generate_scattered_positions(
            num_points=200,
            rng=rng,
        )
        coeffs = numpy.array([2.0, -3.0, 0.5])
        values = _evaluate_linear_sarray(
            positions=positions,
            coeffs=coeffs,
            offset=7.0,
        )
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
        )
        self.assertEqual(
            gradient.shape,
            (200, 3),
        )
        expected = numpy.tile(
            coeffs,
            (200, 1),
        )
        numpy.testing.assert_allclose(
            gradient,
            expected,
            atol=1e-8,
        )

    def test_recovers_constant_gradient_on_clustered_points(
        self,
    ):
        """Point spacing spans orders of magnitude; WLS must stay exact regardless."""
        rng = numpy.random.default_rng(seed=1)
        positions = _generate_clustered_positions(
            num_points_dense=100,
            num_points_sparse=100,
            rng=rng,
        )
        coeffs = numpy.array([1.0, 1.0, 1.0])
        values = _evaluate_linear_sarray(
            positions=positions,
            coeffs=coeffs,
            offset=0.0,
        )
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
        )
        expected = numpy.tile(
            coeffs,
            (positions.shape[0], 1),
        )
        numpy.testing.assert_allclose(
            gradient,
            expected,
            atol=1e-6,
        )


class TestGradientWLS_LinearVectorField(unittest.TestCase):

    def test_recovers_constant_jacobian_on_scattered_points(
        self,
    ):
        rng = numpy.random.default_rng(seed=2)
        positions = _generate_scattered_positions(
            num_points=200,
            rng=rng,
        )
        coeff_matrix = rng.uniform(
            -2.0,
            2.0,
            size=(3, 3),
        )
        offset = rng.uniform(
            -1.0,
            1.0,
            size=3,
        )
        values = _evaluate_linear_varray(
            positions=positions,
            coeff_matrix=coeff_matrix,
            offset=offset,
        )
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
        )
        self.assertEqual(
            gradient.shape,
            (200, 3, 3),
        )
        expected = numpy.tile(
            coeff_matrix,
            (200, 1, 1),
        )
        numpy.testing.assert_allclose(
            gradient,
            expected,
            atol=1e-8,
        )


class TestGradientWLS_Parameters(unittest.TestCase):

    def test_default_k_neighbors_and_weight_power(
        self,
    ):
        rng = numpy.random.default_rng(seed=3)
        positions = _generate_scattered_positions(
            num_points=50,
            rng=rng,
        )
        values = _evaluate_linear_sarray(
            positions=positions,
            coeffs=numpy.array([1.0, 0.0, 0.0]),
            offset=0.0,
        )
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
        )
        self.assertEqual(
            gradient.shape,
            (50, 3),
        )

    def test_smaller_k_neighbors_still_exact_for_linear_field(
        self,
    ):
        rng = numpy.random.default_rng(seed=4)
        positions = _generate_scattered_positions(
            num_points=50,
            rng=rng,
        )
        coeffs = numpy.array([0.0, 4.0, -1.0])
        values = _evaluate_linear_sarray(
            positions=positions,
            coeffs=coeffs,
            offset=0.0,
        )
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=4,
        )
        expected = numpy.tile(
            coeffs,
            (50, 1),
        )
        numpy.testing.assert_allclose(
            gradient,
            expected,
            atol=1e-6,
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } U-TEST
