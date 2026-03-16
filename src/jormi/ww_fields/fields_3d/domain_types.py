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
from typing import cast

## third-party
import numpy

## local
from jormi.ww_fields import _domain_types
from jormi.ww_types import check_types

##
## === 3D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain_3D(_domain_types.UniformDomain):
    """
    Uniform 3D domain: `num_sdims == 3`.

    Fields
    ---
    - `periodicity`:
        Per-axis periodicity flags as (P0, P1, P2).

    - `resolution`:
        Number of cells along each axis as (N0, N1, N2).

    - `domain_bounds`:
        Physical bounds as ((x0_min, x0_max), (x1_min, x1_max), (x2_min, x2_max)).
    """

    num_sdims: int = field(default=3, init=False)
    periodicity: tuple[bool, bool, bool]
    resolution: tuple[int, int, int]
    domain_bounds: tuple[
        tuple[float, float],
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
    ) -> tuple[float, float, float]:
        return cast(
            tuple[float, float, float],
            super().cell_widths,
        )

    @cached_property
    def domain_lengths(
        self,
    ) -> tuple[float, float, float]:
        return cast(
            tuple[float, float, float],
            super().domain_lengths,
        )

    @cached_property
    def cell_centers(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        return cast(
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
            super().cell_centers,
        )

    @cached_property
    def cell_volume(
        self,
    ) -> float:
        return float(super()._measure_per_cell)

    @cached_property
    def total_volume(
        self,
    ) -> float:
        return float(super()._total_measure)


##
## === TYPE VALIDATION
##


def ensure_3d_udomain(
    udomain_3d: UniformDomain_3D,
    *,
    param_name: str = "<udomain_3d>",
) -> None:
    check_types.ensure_type(
        param=udomain_3d,
        param_name=param_name,
        valid_types=UniformDomain_3D,
    )


def ensure_3d_periodic_udomain(
    udomain_3d: UniformDomain_3D,
    *,
    param_name: str = "<udomain_3d>",
) -> None:
    """
    Ensure `udomain_3d` is a UniformDomain_3D that is periodic in all directions.

    Intended for FFT-based operations (e.g. Helmholtz decomposition) that assume
    fully periodic boundary conditions.
    """
    ensure_3d_udomain(
        udomain_3d=udomain_3d,
        param_name=param_name,
    )
    if not all(udomain_3d.periodicity):
        raise ValueError(
            f"{param_name} must be periodic in all directions for this operation;"
            f" periodicity={udomain_3d.periodicity}.",
        )


## } MODULE