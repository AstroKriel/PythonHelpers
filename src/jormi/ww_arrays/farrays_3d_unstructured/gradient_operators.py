## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.spatial import KDTree as scipy_KDTree

##
## === CONSTANTS
##

_DISTANCE_EPS: float = 1e-30

##
## === GRADIENT OPERATORS
##


def compute_gradient_wls(
    positions: NDArray[Any],
    values: NDArray[Any],
    *,
    k_neighbors: int = 20,
    weight_power: float = 2,
) -> NDArray[numpy.float64]:
    """
    Compute the gradient of a scalar or vector field on an unstructured point cloud.

    Uses weighted least-squares (WLS) with inverse-distance weights. Each cell's
    gradient is estimated from the `k_neighbors` nearest neighbours by fitting a
    linear model to the displaced values. Works for any particle or cell data,
    including Voronoi mesh outputs from Arepo or SPH particle data.

    Parameters
    ---
    - `positions`:
        Cell or particle centroids; shape (N, 3).

    - `values`:
        Field values; shape (N,) for a scalar or (N, M) for an M-component vector.

    - `k_neighbors`:
        Number of nearest neighbours (excluding self) used in the WLS fit.

    - `weight_power`:
        Exponent for inverse-distance weighting: `w = 1 / d^weight_power`.

    Returns
    ---
    - `gradient`:
        Shape (N, 3) for a scalar input or (N, M, 3) for a vector input.
        `gradient[n, d]` is `d_d f` at cell n; `gradient[n, m, d]` is `d_d f_m`.
    """
    positions_64 = numpy.asarray(positions, dtype=numpy.float64)
    values_64 = numpy.asarray(values, dtype=numpy.float64)
    is_scalar = values_64.ndim == 1
    if is_scalar:
        values_64 = values_64[:, numpy.newaxis]
    n_cells, n_components = values_64.shape
    tree = scipy_KDTree(positions_64)
    neighbor_distances, neighbor_indices = tree.query(positions_64, k=k_neighbors + 1)
    ## drop the first column (self, distance = 0) and keep only k_neighbors neighbours
    neighbor_distances = neighbor_distances[:, 1:]
    neighbor_indices = neighbor_indices[:, 1:]
    ## displacements from each cell to its neighbours; shape (N, k, 3)
    displacements = positions_64[neighbor_indices] - positions_64[:, numpy.newaxis, :]
    ## inverse-distance weights; shape (N, k)
    weights = 1.0 / numpy.maximum(neighbor_distances, _DISTANCE_EPS) ** weight_power
    ## normal equations matrix M_ij = sum_k w_k dx_ki dx_kj; shape (N, 3, 3)
    normal_matrix = numpy.einsum("nk,nki,nkj->nij", weights, displacements, displacements)
    ## right-hand side; value differences weighted by w; shape (N, k, M)
    value_differences = values_64[neighbor_indices] - values_64[:, numpy.newaxis, :]
    ## rhs shape (N, 3, M): sum_k w_k dx_ki (f_k - f_0)
    rhs = numpy.einsum("nk,nki,nkm->nim", weights, displacements, value_differences)
    ## batch linear solve: M g = rhs; result shape (N, 3, M)
    gradient_components = numpy.linalg.solve(normal_matrix, rhs)
    if is_scalar:
        return gradient_components[:, :, 0]
    ## reorder from (N, 3, M) to (N, M, 3)
    return gradient_components.transpose(0, 2, 1)


## } MODULE
