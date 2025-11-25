## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass
from typing import cast

from jormi.ww_types import type_manager
from jormi.ww_fields import _domain_types


##
## === 2D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain_2D(_domain_types.UniformDomain):
    """
    2D uniform domain: num_sdims == 2.

    - periodicity: (px, py)
    - resolution:  (Nx, Ny)
    - domain_bounds: ((x_min, x_max), (y_min, y_max))
    """

    periodicity: tuple[bool, bool]
    resolution: tuple[int, int]
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
    ]

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(self, "num_sdims", 2)
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
## === TYPE VALIDATION
##


def ensure_2d_udomain(
    udomain_2d: UniformDomain_2D,
    *,
    param_name: str = "<udomain>",
) -> None:
    type_manager.ensure_type(
        param=udomain_2d,
        param_name=param_name,
        valid_types=UniformDomain_2D,
    )


## } MODULE
