## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import type_manager, cartesian_coordinates

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class UniformDomain:
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
        self._validate_periodicity()
        self._validate_resolution()
        self._validate_domain_bounds()

    def _validate_periodicity(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.periodicity,
            param_name="<periodicity>",
            seq_length=3,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Booleans.BooleanLike,
        )

    def _validate_resolution(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.resolution,
            param_name="<resolution>",
            seq_length=3,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Numerics.IntLike,
        )
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        if (num_cells_x <= 0) or (num_cells_y <= 0) or (num_cells_z <= 0):
            raise ValueError("All `<resolution>` entries must be positive.")

    def _validate_domain_bounds(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.domain_bounds,
            param_name="<domain_bounds>",
            seq_length=3,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Sequences.TupleLike,
        )
        for axis_index, bounds in enumerate(self.domain_bounds):
            axis_label = cartesian_coordinates.DEFAULT_AXES_ORDER[axis_index].value
            axis_param_name = f"<domain_bounds[{axis_label}]>"
            type_manager.ensure_sequence(
                param=bounds,
                param_name=axis_param_name,
                seq_length=2,
                valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
                valid_elem_types=type_manager.RuntimeTypes.Numerics.NumericLike,
            )
            lo_value, hi_value = bounds
            type_manager.ensure_finite_float(
                param=lo_value,
                param_name=f"{axis_param_name}[0]",
                allow_none=False,
                require_positive=False,
            )
            type_manager.ensure_finite_float(
                param=hi_value,
                param_name=f"{axis_param_name}[1]",
                allow_none=False,
                require_positive=False,
            )
            if not (hi_value > lo_value):
                raise ValueError(f"{axis_label}-axis: max bound must be > min bound.")

    @cached_property
    def cell_widths(
        self,
    ) -> tuple[float, float, float]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        return (
            (x_max - x_min) / num_cells_x,
            (y_max - y_min) / num_cells_y,
            (z_max - z_min) / num_cells_z,
        )

    @cached_property
    def cell_volume(
        self,
    ) -> float:
        return float(numpy.prod(self.cell_widths))

    @cached_property
    def domain_lengths(
        self,
    ) -> tuple[float, float, float]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        return (
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )

    @cached_property
    def num_cells(
        self,
    ) -> int:
        return int(numpy.prod(self.resolution))

    @cached_property
    def total_volume(
        self,
    ) -> float:
        return float(numpy.prod(self.domain_lengths))

    @cached_property
    def cell_centers(
        self,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:

        def _get_bin_centers(
            axis_min: float,
            cell_width: float,
            num_cells: int,
        ) -> numpy.ndarray:
            return axis_min + (numpy.arange(num_cells, dtype=float) + 0.5) * cell_width

        (x_min, _), (y_min, _), (z_min, _) = self.domain_bounds
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        cell_width_x, cell_width_y, cell_width_z = self.cell_widths
        x_centers = _get_bin_centers(x_min, cell_width_x, num_cells_x)
        y_centers = _get_bin_centers(y_min, cell_width_y, num_cells_y)
        z_centers = _get_bin_centers(z_min, cell_width_z, num_cells_z)
        return (
            x_centers,
            y_centers,
            z_centers,
        )


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
