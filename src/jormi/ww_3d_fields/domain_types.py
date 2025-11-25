## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass
from typing import cast

from jormi.ww_types import (
    type_manager,
    domain_types as base_domain_types,
)

##
## === 3D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain(base_domain_types.UniformDomain):
    """
    Uniform 3D domain.

    - periodicity: (Px, Py, Pz)
    - resolution:  (Nx, Ny, Nz)
    - domain_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """

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
        object.__setattr__(self, "num_sdims", 3)
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


def ensure_udomain(
    udomain: UniformDomain,
    *,
    param_name: str = "<udomain>",
) -> None:
    type_manager.ensure_type(
        param=udomain,
        param_name=param_name,
        valid_types=UniformDomain,
    )

## } MODULE
