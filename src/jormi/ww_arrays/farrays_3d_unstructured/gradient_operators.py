## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, Literal, overload

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


@overload
def compute_gradient_wls(
    positions: NDArray[Any],
    values: NDArray[Any],
    *,
    k_neighbors: int = 20,
    weight_power: float = 2,
    min_distances: NDArray[Any] | None = None,
    return_floor_mask: Literal[False] = False,
) -> NDArray[numpy.float64]: ...


@overload
def compute_gradient_wls(
    positions: NDArray[Any],
    values: NDArray[Any],
    *,
    k_neighbors: int = 20,
    weight_power: float = 2,
    min_distances: NDArray[Any] | None = None,
    return_floor_mask: Literal[True],
) -> tuple[NDArray[numpy.float64], NDArray[numpy.bool_]]: ...


def compute_gradient_wls(
    positions: NDArray[Any],
    values: NDArray[Any],
    *,
    k_neighbors: int = 20,
    weight_power: float = 2,
    min_distances: NDArray[Any] | None = None,
    return_floor_mask: bool = False,
) -> NDArray[numpy.float64] | tuple[NDArray[numpy.float64], NDArray[numpy.bool_]]:
    """
    Compute the gradient of a scalar or vector field on an unstructured point cloud.

    Uses weighted least-squares (WLS) with inverse-distance weights. Each cell's
    gradient is estimated from the `k_neighbors` nearest neighbours by fitting a
    linear model to the displaced values. Works for any particle or cell data,
    including Voronoi mesh outputs from Arepo or SPH particle data.

    Two (near-)coincident points make the `1/d^weight_power` weighting blow up: a
    modest value difference over a near-zero separation is read as an enormous
    gradient. `min_distances` guards against this by refusing to trust a neighbour
    separation below each cell's own resolvable scale.

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

    - `min_distances`:
        Per-cell floor on neighbour distance used for weighting; shape (N,). A
        neighbour closer than `min_distances[n]` is weighted as if it were exactly
        that far away. `None` (default) applies no floor, matching prior behaviour
        exactly. A natural choice is each cell's own size, e.g. `volumes ** (1 / 3)`
        for a `PointCloudDomain`: a gradient can't be resolved below a cell's own scale.

    - `return_floor_mask`:
        If `True`, also return a boolean mask of shape (N,) marking cells whose
        nearest neighbour was closer than `min_distances` and therefore floored, so
        callers can audit how often the guard actually engages.

    Returns
    ---
    - `gradient`:
        Shape (N, 3) for a scalar input or (N, M, 3) for a vector input.
        `gradient[n, d]` is `d_d f` at cell n; `gradient[n, m, d]` is `d_d f_m`.

    - `floor_mask` (only if `return_floor_mask` is `True`):
        Shape (N,); `True` where the nearest neighbour distance was floored.
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
    if min_distances is not None:
        min_distances_64 = numpy.asarray(min_distances, dtype=numpy.float64)
        floor_mask = neighbor_distances[:, 0] < min_distances_64
        neighbor_distances = numpy.maximum(neighbor_distances, min_distances_64[:, numpy.newaxis])
    else:
        floor_mask = numpy.zeros(n_cells, dtype=numpy.bool_)
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
        gradient = gradient_components[:, :, 0]
    else:
        ## reorder from (N, 3, M) to (N, M, 3)
        gradient = gradient_components.transpose(0, 2, 1)
    if return_floor_mask:
        return gradient, floor_mask
    return gradient


## } MODULE
