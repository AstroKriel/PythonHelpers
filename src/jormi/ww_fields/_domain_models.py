## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from functools import cached_property
from typing import Any

## third-party
import numpy
from numpy.typing import NDArray
from scipy.spatial import KDTree as scipy_KDTree

## local
from jormi.ww_fields import cartesian_axes
from jormi.ww_validation import validate_types

##
## === ABSTRACT BASE
##


@dataclass(frozen=True)
class Domain(ABC):
    """
    Abstract base for a spatial domain: dimensionality, periodicity, and physical extent.

    Concrete domains describe cells differently (a fixed per-axis grid vs. an
    unstructured point cloud), so `resolution`, `cell_widths`, and similar
    grid-specific concepts live on `UniformDomain`, not here.

    Fields
    ---
    - `num_sdims`:
        Number of spatial dimensions.

    - `periodicity`:
        Per-axis periodicity flags; length must equal `num_sdims`.

    - `domain_bounds`:
        Physical (min, max) bounds for each axis; length must equal `num_sdims`.
    """

    num_sdims: int
    periodicity: tuple[bool, ...]
    domain_bounds: tuple[tuple[float, float], ...]

    def __post_init__(
        self,
    ) -> None:
        ## validate the per-axis domain metadata
        self._ensure_num_sdims()
        self._ensure_periodicity()
        self._ensure_domain_bounds()

    def _axis_label_from_index(
        self,
        *,
        axis_index: int,
    ) -> str:
        return cartesian_axes.get_axis_label(axis_index)

    def _ensure_num_sdims(
        self,
    ) -> None:
        validate_types.ensure_finite_int(
            param=self.num_sdims,
            param_name="<num_sdims>",
            allow_none=False,
            allow_zero=False,
            require_positive=True,
        )

    def _ensure_periodicity(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.periodicity,
            param_name="<periodicity>",
            seq_length=self.num_sdims,
            valid_seq_types=validate_types.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=validate_types.RuntimeTypes.Booleans.BooleanLike,
        )

    def _ensure_domain_bounds(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.domain_bounds,
            param_name="<domain_bounds>",
            seq_length=self.num_sdims,
            valid_seq_types=validate_types.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=validate_types.RuntimeTypes.Sequences.TupleLike,
        )
        for axis_index in range(self.num_sdims):
            bounds = self.domain_bounds[axis_index]
            axis_label = self._axis_label_from_index(
                axis_index=axis_index,
            )
            axis_param_name = f"<domain_bounds[{axis_label}]>"
            validate_types.ensure_sequence(
                param=bounds,
                param_name=axis_param_name,
                seq_length=2,
                valid_seq_types=validate_types.RuntimeTypes.Sequences.TupleLike,
                valid_elem_types=validate_types.RuntimeTypes.Numerics.NumericLike,
            )
            lo_value, hi_value = bounds
            validate_types.ensure_finite_float(
                param=lo_value,
                param_name=f"{axis_param_name}[0]",
                allow_none=False,
                allow_zero=True,
                require_positive=False,
            )
            validate_types.ensure_finite_float(
                param=hi_value,
                param_name=f"{axis_param_name}[1]",
                allow_none=False,
                allow_zero=True,
                require_positive=False,
            )
            if not (hi_value > lo_value):
                raise ValueError(
                    f"{axis_label}-axis: max bound must be > min bound.",
                )

    @cached_property
    def domain_lengths(
        self,
    ) -> tuple[float, ...]:
        return tuple(axis_bounds[1] - axis_bounds[0] for axis_bounds in self.domain_bounds)

    @property
    @abstractmethod
    def num_cells(
        self,
    ) -> int:
        """Total number of cells or points described by this domain."""
        ...

    @property
    @abstractmethod
    def expected_sdims_shape(
        self,
    ) -> tuple[int, ...]:
        """Spatial shape a `FieldData` must have to be sampled on this domain."""
        ...


##
## === UNIFORM GRID DOMAIN
##


@dataclass(frozen=True)
class UniformDomain(Domain):
    """
    A uniform Cartesian domain: a fixed number of equal-sized cells per axis.

    Fields
    ---
    - `resolution`:
        Number of cells along each axis; length must equal `num_sdims`.
    """

    resolution: tuple[int, ...]

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._ensure_resolution()

    def _ensure_resolution(
        self,
    ) -> None:
        validate_types.ensure_sequence(
            param=self.resolution,
            param_name="<resolution>",
            seq_length=self.num_sdims,
            valid_seq_types=validate_types.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=validate_types.RuntimeTypes.Numerics.IntLike,
        )
        for axis_index, num_cells in enumerate(self.resolution):
            axis_label = self._axis_label_from_index(
                axis_index=axis_index,
            )
            if num_cells <= 0:
                raise ValueError(
                    f"`<resolution>[{axis_label}]` must be a positive integer.",
                )

    @cached_property
    def cell_widths(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            (axis_bounds[1] - axis_bounds[0]) / num_cells
            for axis_bounds, num_cells in zip(self.domain_bounds, self.resolution)
        )

    @property
    def num_cells(
        self,
    ) -> int:
        return int(
            numpy.prod(
                self.resolution,
            ),
        )

    @property
    def expected_sdims_shape(
        self,
    ) -> tuple[int, ...]:
        return self.resolution

    @cached_property
    def _measure_per_cell(
        self,
    ) -> float:
        """Area per cell if 2D; volume per cell if 3D."""
        return float(
            numpy.prod(
                self.cell_widths,
            ),
        )

    @cached_property
    def _total_measure(
        self,
    ) -> float:
        """Total area if 2D; total volume if 3D."""
        return float(
            numpy.prod(
                self.domain_lengths,
            ),
        )

    @cached_property
    def cell_centers(
        self,
    ) -> tuple[NDArray[Any], ...]:

        def _get_cell_centers(
            *,
            axis_min: float,
            cell_width: float,
            num_cells: int,
        ) -> NDArray[Any]:
            return axis_min + (numpy.arange(num_cells, dtype=float) + 0.5) * cell_width

        cell_centers_per_axis: list[NDArray[Any]] = []
        for (axis_min, _), cell_width, num_cells in zip(
                self.domain_bounds,
                self.cell_widths,
                self.resolution,
        ):
            cell_centers = _get_cell_centers(
                axis_min=axis_min,
                cell_width=cell_width,
                num_cells=num_cells,
            )
            cell_centers_per_axis.append(cell_centers)
        return tuple(cell_centers_per_axis)


##
## === POINT-CLOUD DOMAIN
##


@dataclass(frozen=True)
class PointCloudDomain(Domain):
    """
    An unstructured domain: cells or particles at arbitrary positions, e.g. a Voronoi
    mesh (Arepo) or SPH particles. Cell size is not fixed, so there is no `resolution`
    or `cell_widths`; `positions` is the data that cannot be derived from smaller
    parameters, the same role `resolution` plays for `UniformDomain`.

    Fields
    ---
    - `positions`:
        Cell or particle centroids; shape (num_cells, num_sdims).

    - `volumes`:
        Per-cell volume (area in 2D); shape (num_cells,); `None` if not available.
        This is domain geometry (how much space each cell occupies, shared by every
        field sampled on this domain), not a physical field like mass or density.
    """

    positions: NDArray[Any]
    volumes: NDArray[Any] | None = None

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._ensure_positions()
        self._ensure_volumes()

    def _ensure_positions(
        self,
    ) -> None:
        validate_types.ensure_type(
            param=self.positions,
            param_name="<positions>",
            valid_types=numpy.ndarray,
        )
        if self.positions.ndim != 2:
            raise ValueError(
                f"`<positions>` must have shape (num_cells, {self.num_sdims});"
                f" got ndim={self.positions.ndim}.",
            )
        if self.positions.shape[1] != self.num_sdims:
            raise ValueError(
                f"`<positions>` must have shape (num_cells, {self.num_sdims});"
                f" got shape={self.positions.shape}.",
            )

    def _ensure_volumes(
        self,
    ) -> None:
        if self.volumes is None:
            return
        validate_types.ensure_type(
            param=self.volumes,
            param_name="<volumes>",
            valid_types=numpy.ndarray,
        )
        if self.volumes.shape != (self.positions.shape[0], ):
            raise ValueError(
                f"`<volumes>` must have shape ({self.positions.shape[0]},);"
                f" got shape={self.volumes.shape}.",
            )

    @property
    def num_cells(
        self,
    ) -> int:
        return int(self.positions.shape[0])

    @property
    def expected_sdims_shape(
        self,
    ) -> tuple[int, ...]:
        return (self.num_cells, )

    @cached_property
    def total_cell_volume(
        self,
    ) -> float:
        """Sum of per-cell volumes; requires `volumes`, not derived from `domain_bounds`."""
        if self.volumes is None:
            raise ValueError("`<volumes>` is None; cannot compute `total_cell_volume`.")
        return float(self.volumes.sum())

    @cached_property
    def kdtree(
        self,
    ) -> scipy_KDTree:
        """Spatial index on `positions`, built once and reused by point-cloud operators."""
        return scipy_KDTree(self.positions)


##
## === TYPE VALIDATION
##


def ensure_uniform_domain(
    uniform_domain: Domain,
    *,
    param_name: str = "<uniform_domain>",
) -> None:
    validate_types.ensure_type(
        param=uniform_domain,
        param_name=param_name,
        valid_types=Domain,
    )


def ensure_uniform_domain_metadata(
    uniform_domain: Domain,
    *,
    num_sdims: int | None = None,
    param_name: str = "<uniform_domain>",
) -> None:
    """Check metadata for a `Domain`."""
    ensure_uniform_domain(
        uniform_domain=uniform_domain,
        param_name=param_name,
    )
    if (num_sdims is not None) and (uniform_domain.num_sdims != num_sdims):
        raise ValueError(
            f"`{param_name}` must have num_sdims={num_sdims},"
            f" but got num_sdims={uniform_domain.num_sdims}.",
        )


## } MODULE
