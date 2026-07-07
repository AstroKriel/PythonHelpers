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


class TestGradientWLS_DistanceFloor(unittest.TestCase):
    """
    Two (near-)coincident points make `1/d^weight_power` blow up: a modest value
    difference over a near-zero separation reads as an enormous gradient. This
    mirrors real Arepo cells found at essentially coincident centroids with
    genuinely different velocities (see `<project-notes>/research/lead-author/
    kriel-whittingham-galactic-dynamos/threads/dynamo-gradients/README.md`).
    """

    def _generate_field_with_coincident_pair(
        self,
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
        rng = numpy.random.default_rng(seed=42)
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
        ## a near-duplicate of point 5 with a value inconsistent with the smooth field,
        ## mimicking two distinct Arepo cells that happen to sit at the same position
        duplicate_position = positions[5] + 1e-10
        duplicate_value = values[5] + 500.0
        positions_with_duplicate = numpy.concatenate(
            [positions, duplicate_position[numpy.newaxis, :]],
            axis=0,
        )
        values_with_duplicate = numpy.concatenate(
            [values, [duplicate_value]],
        )
        return positions_with_duplicate, values_with_duplicate, coeffs

    def test_coincident_pair_explodes_without_floor(
        self,
    ):
        positions, values, _coeffs = self._generate_field_with_coincident_pair()
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
        )
        self.assertGreater(
            numpy.linalg.norm(gradient[5]),
            1e6,
        )

    def test_floor_recovers_clean_neighbours_gradient(
        self,
    ):
        """The floor should let cell 5, whose own data is clean, recover its true gradient."""
        positions, values, coeffs = self._generate_field_with_coincident_pair()
        min_distances = numpy.full(positions.shape[0], 0.01)
        gradient = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
            min_distances=min_distances,
        )
        numpy.testing.assert_allclose(
            gradient[5],
            coeffs,
            atol=1e-2,
        )
        ## the duplicate's own gradient is not required to recover `coeffs` (its own value
        ## really is anomalous relative to its neighbours), but must stay bounded, not explode
        self.assertLess(
            numpy.linalg.norm(gradient[-1]),
            1e4,
        )

    def test_floor_mask_flags_only_the_coincident_pair(
        self,
    ):
        positions, values, _coeffs = self._generate_field_with_coincident_pair()
        min_distances = numpy.full(positions.shape[0], 0.01)
        _gradient, floor_mask = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
            min_distances=min_distances,
            return_floor_mask=True,
        )
        self.assertEqual(
            floor_mask.shape,
            (positions.shape[0],),
        )
        numpy.testing.assert_array_equal(
            numpy.where(floor_mask)[0],
            numpy.array([5, positions.shape[0] - 1]),
        )

    def test_none_min_distances_matches_no_floor_argument(
        self,
    ):
        """Passing `min_distances=None` explicitly must be identical to omitting it."""
        positions, values, _coeffs = self._generate_field_with_coincident_pair()
        gradient_omitted = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
        )
        gradient_explicit_none = gradient_operators.compute_gradient_wls(
            positions,
            values,
            k_neighbors=20,
            min_distances=None,
        )
        numpy.testing.assert_array_equal(
            gradient_omitted,
            gradient_explicit_none,
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
