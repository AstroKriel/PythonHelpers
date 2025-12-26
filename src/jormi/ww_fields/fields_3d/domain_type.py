## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import cast
from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import type_checks
from jormi.ww_fields import _domain_type

##
## === 3D DOMAIN
##


@dataclass(frozen=True)
class UniformDomain_3D(_domain_type.UniformDomain):
    """
    Uniform 3D domain: `num_sdims == 3`.

    - `periodicity`: (Px, Py, Pz)
    - `resolution`:  (Nx, Ny, Nz)
    - `domain_bounds`: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """

    periodicity: tuple[bool, bool, bool]
    resolution: tuple[int, int, int]
    domain_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    def __init__(
        self,
        periodicity: tuple[bool, bool, bool],
        resolution: tuple[int, int, int],
        domain_bounds: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> None:
        super().__init__(
            num_sdims=3,
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
    type_checks.ensure_type(
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
