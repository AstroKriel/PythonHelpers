## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import cast
from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import type_manager
from jormi.ww_fields import (
    _domains,
    _cartesian_coordinates,
)

##
## === 2D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain_2D(_domains.UniformDomain):
    """
    Uniform 2D domain: `num_sdims == 2`.

    - `periodicity`: (Px, Py)
    - `resolution`:  (Nx, Ny)
    - `domain_bounds`: ((x_min, x_max), (y_min, y_max))
    """

    periodicity: tuple[bool, bool]
    resolution: tuple[int, int]
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
    ]

    def __init__(
        self,
        *,
        periodicity: tuple[bool, bool],
        resolution: tuple[int, int],
        domain_bounds: tuple[
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> None:
        ## fix num_sdims for all 2D domains and forward metadata to the base class
        super().__init__(
            num_sdims=2,
            periodicity=periodicity,
            resolution=resolution,
            domain_bounds=domain_bounds,
        )

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
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        return cast(
            tuple[numpy.ndarray, numpy.ndarray],
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

    Behaves like a regular 2D uniform domain, but carries extra
    metadata about its 3D origin:

    - `out_of_plane_axis`: out-of-plane axis removed from the original 3D domain (x, y, or z).
    - `slice_index`: grid index along `out_of_plane_axis` where the slice was taken.
    - `slice_position`: physical coordinate along `out_of_plane_axis` where the slice was taken.
    """

    out_of_plane_axis: _cartesian_coordinates.AxisLike
    slice_index: int
    slice_position: float

    def __init__(
        self,
        *,
        periodicity: tuple[bool, bool],
        resolution: tuple[int, int],
        domain_bounds: tuple[
            tuple[float, float],
            tuple[float, float],
        ],
        out_of_plane_axis: _cartesian_coordinates.AxisLike,
        slice_index: int,
        slice_position: float,
    ) -> None:
        ## construct the underlying 2D domain (num_sdims is fixed to 2 there)
        super().__init__(
            periodicity=periodicity,
            resolution=resolution,
            domain_bounds=domain_bounds,
        )
        ## assign extra sliced-3D metadata on the frozen dataclass
        object.__setattr__(self, "out_of_plane_axis", out_of_plane_axis)
        object.__setattr__(self, "slice_index", slice_index)
        object.__setattr__(self, "slice_position", slice_position)

    def __post_init__(
        self,
    ) -> None:
        ## validate the underlying 2D domain itself
        super().__post_init__()
        ## normalise and store the axis as a CartesianAxis enum
        slice_axis_enum = _cartesian_coordinates.as_axis(
            axis=self.out_of_plane_axis,
            param_name="out_of_plane_axis",
        )
        object.__setattr__(self, "out_of_plane_axis", slice_axis_enum)
        ## validate index and coordinate
        type_manager.ensure_finite_int(
            param=self.slice_index,
            param_name="slice_index",
            allow_none=False,
            require_positive=False,
        )
        if self.slice_index < 0:
            raise ValueError(
                "slice_index must be non-negative:"
                f" got {self.slice_index}",
            )
        type_manager.ensure_finite_float(
            param=self.slice_position,
            param_name="slice_position",
            allow_none=False,
        )

    @cached_property
    def sliced_axis_index(
        self,
    ) -> int:
        """Default integer index (0, 1, 2) of the out-of-plane slice axis."""
        return _cartesian_coordinates.get_axis_index(self.out_of_plane_axis)


##
## === TYPE VALIDATION
##


def ensure_2d_udomain(
    udomain_2d: UniformDomain_2D,
    *,
    param_name: str = "<udomain_2d>",
) -> None:
    type_manager.ensure_type(
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
    type_manager.ensure_type(
        param=udomain_2d,
        param_name=param_name,
        valid_types=UniformDomain_2D_Sliced3D,
    )


## } MODULE
