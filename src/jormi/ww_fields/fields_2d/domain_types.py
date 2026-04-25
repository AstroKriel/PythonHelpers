## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import (
    dataclass,
    field,
)
from functools import cached_property
from typing import Any, cast

## third-party
from numpy.typing import NDArray

## local
from jormi.ww_fields import (
    _domain_types,
    cartesian_axes,
)
from jormi.ww_validation import validate_types

##
## === 2D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain_2D(_domain_types.UniformDomain):
    """
    Uniform 2D domain: `num_sdims == 2`.

    Fields
    ---
    - `periodicity`:
        Per-axis periodicity flags as (is_x0_periodic, is_x1_periodic).

    - `resolution`:
        Number of cells along each axis as (num_x0_cells, num_x1_cells).

    - `domain_bounds`:
        Physical bounds as ((x0_min, x0_max), (x1_min, x1_max)).
    """

    num_sdims: int = field(default=2, init=False)
    periodicity: tuple[bool, bool]
    resolution: tuple[int, int]
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
    ]

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()

    @cached_property
    def cell_widths(
        self,
    ) -> tuple[float, float]:
        return cast(
            tuple[float, float],
            super().cell_widths,
        )

    @cached_property
    def domain_lengths(
        self,
    ) -> tuple[float, float]:
        return cast(
            tuple[float, float],
            super().domain_lengths,
        )

    @cached_property
    def cell_centers(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        return cast(
            tuple[NDArray[Any], NDArray[Any]],
            super().cell_centers,
        )

    @cached_property
    def cell_area(
        self,
    ) -> float:
        return float(super()._measure_per_cell)

    @cached_property
    def total_area(
        self,
    ) -> float:
        return float(super()._total_measure)


##
## === 2D DOMAIN SLICED FROM 3D
##


@dataclass(frozen=True)
class UniformDomain_2D_Sliced3D(UniformDomain_2D):
    """
    2D uniform domain obtained by slicing a 3D uniform domain.

    Behaves like a regular 2D uniform domain, but carries extra metadata:

    - `out_of_plane_axis`: axis removed from the original 3D domain (stored canonically).
    - `slice_index`: grid index along `out_of_plane_axis` where the slice was taken.
    - `slice_position`: physical coordinate along `out_of_plane_axis` where the slice was taken.
    """

    out_of_plane_axis: cartesian_axes.CartesianAxis_3D
    slice_index: int
    slice_position: float

    @classmethod
    def from_slice(
        cls,
        *,
        periodicity: tuple[bool, bool],
        resolution: tuple[int, int],
        domain_bounds: tuple[
            tuple[float, float],
            tuple[float, float],
        ],
        out_of_plane_axis: cartesian_axes.AxisLike_3D,
        slice_index: int,
        slice_position: float,
    ) -> "UniformDomain_2D_Sliced3D":
        out_of_plane_axis = cartesian_axes.as_axis(
            axis=out_of_plane_axis,
            param_name="out_of_plane_axis",
        )
        return cls(
            periodicity=periodicity,
            resolution=resolution,
            domain_bounds=domain_bounds,
            out_of_plane_axis=out_of_plane_axis,
            slice_index=slice_index,
            slice_position=slice_position,
        )

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        validate_types.ensure_type(
            param=self.out_of_plane_axis,
            param_name="out_of_plane_axis",
            valid_types=cartesian_axes.CartesianAxis_3D,
        )
        validate_types.ensure_finite_int(
            param=self.slice_index,
            param_name="slice_index",
            allow_none=False,
            allow_zero=True,
            require_positive=True,
        )
        if self.slice_index < 0:
            raise ValueError(
                "slice_index must be non-negative:"
                f" got {self.slice_index}",
            )
        validate_types.ensure_finite_float(
            param=self.slice_position,
            param_name="slice_position",
            allow_none=False,
            allow_zero=True,
            require_positive=False,
        )

    @cached_property
    def sliced_axis_index(
        self,
    ) -> cartesian_axes.AxisIndex_3D:
        """Integer index (0, 1, 2) of the out-of-plane slice axis."""
        return self.out_of_plane_axis.axis_index


##
## === TYPE VALIDATION
##


def ensure_2d_udomain(
    udomain_2d: UniformDomain_2D,
    *,
    param_name: str = "<udomain_2d>",
) -> None:
    validate_types.ensure_type(
        param=udomain_2d,
        param_name=param_name,
        valid_types=UniformDomain_2D,
    )


def ensure_2d_udomain_sliced_from_3d(
    udomain_2d: UniformDomain_2D,
    *,
    param_name: str = "<udomain_2d>",
) -> None:
    """
    Ensure `udomain_2d` is a 2D domain sliced from 3D.

    Note:
    We accept `UniformDomain_2D` in the signature so static type annotation
    understands `field.udomain` (which is annotated as `UniformDomain_2D`);
    the actual subtype-check is enforced via `valid_types=UniformDomain_2D_Sliced3D`.
    """
    validate_types.ensure_type(
        param=udomain_2d,
        param_name=param_name,
        valid_types=UniformDomain_2D_Sliced3D,
    )


## } MODULE
