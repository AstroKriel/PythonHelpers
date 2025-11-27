## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import type_manager
from jormi.ww_fields import _cartesian_coordinates

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class UniformDomain:
    """Base uniform domain."""

    num_sdims: int
    periodicity: tuple[bool, ...]
    resolution: tuple[int, ...]
    domain_bounds: tuple[tuple[float, float], ...]

    def __post_init__(
        self,
    ) -> None:
        self._validate_num_sdims()
        self._validate_periodicity()
        self._validate_resolution()
        self._validate_domain_bounds()

    def _validate_num_sdims(
        self,
    ) -> None:
        type_manager.ensure_finite_int(
            param=self.num_sdims,
            param_name="<num_sdims>",
            allow_none=False,
            require_positive=True,
        )

    def _validate_periodicity(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.periodicity,
            param_name="<periodicity>",
            seq_length=self.num_sdims,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Booleans.BooleanLike,
        )

    def _validate_resolution(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.resolution,
            param_name="<resolution>",
            seq_length=self.num_sdims,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Numerics.IntLike,
        )
        for axis_index, num_cells in enumerate(self.resolution):
            axis_label = _cartesian_coordinates.DEFAULT_AXES_ORDER[axis_index].value
            if num_cells <= 0:
                raise ValueError(
                    f"`<resolution>[{axis_label}]` must be a positive integer.",
                )

    def _validate_domain_bounds(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.domain_bounds,
            param_name="<domain_bounds>",
            seq_length=self.num_sdims,
            valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
            valid_elem_types=type_manager.RuntimeTypes.Sequences.TupleLike,
        )
        for axis_index in range(self.num_sdims):
            bounds = self.domain_bounds[axis_index]
            axis_label = _cartesian_coordinates.DEFAULT_AXES_ORDER[axis_index].value
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
                raise ValueError(
                    f"{axis_label}-axis: max bound must be > min bound.",
                )

    @cached_property
    def cell_widths(
        self,
    ) -> tuple[float, ...]:
        return tuple(
            (axis_bounds[1] - axis_bounds[0]) / num_cells
            for axis_bounds, num_cells in zip(self.domain_bounds, self.resolution)
        )

    @cached_property
    def domain_lengths(
        self,
    ) -> tuple[float, ...]:
        return tuple(axis_bounds[1] - axis_bounds[0] for axis_bounds in self.domain_bounds)

    @cached_property
    def num_cells(
        self,
    ) -> int:
        return int(numpy.prod(self.resolution))

    @cached_property
    def _measure_per_cell(
        self,
    ) -> float:
        """Area per cell in 2D; volume per cell in 3D."""
        return float(numpy.prod(self.cell_widths))

    @cached_property
    def _total_measure(
        self,
    ) -> float:
        """Total area in 2D; total volume in 3D."""
        return float(numpy.prod(self.domain_lengths))

    @cached_property
    def cell_centers(
        self,
    ) -> tuple[numpy.ndarray, ...]:

        def _get_cell_centers(
            axis_min: float,
            cell_width: float,
            num_cells: int,
        ) -> numpy.ndarray:
            return axis_min + (numpy.arange(num_cells, dtype=float) + 0.5) * cell_width

        cell_centers_per_axis: list[numpy.ndarray] = []
        for (axis_min, _), cell_width, num_cells in zip(
                self.domain_bounds,
                self.cell_widths,
                self.resolution,
        ):
            cell_centers = _get_cell_centers(axis_min, cell_width, num_cells)
            cell_centers_per_axis.append(cell_centers)
        return tuple(cell_centers_per_axis)


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


def ensure_udomain_metadata(
    udomain: UniformDomain,
    *,
    num_sdims: int | None = None,
    param_name: str = "<udomain>",
) -> None:
    """Check metadata for `UniformDomain`."""
    ensure_udomain(
        udomain=udomain,
        param_name=param_name,
    )
    if (num_sdims is not None) and (udomain.num_sdims != num_sdims):
        raise ValueError(
            f"`{param_name}` must have num_sdims={num_sdims},"
            f" but got num_sdims={udomain.num_sdims}.",
        )


## } MODULE
