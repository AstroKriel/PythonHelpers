## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass

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

    def __post_init__(self):
        self._validate_periodicity()
        self._validate_resolution()
        self._validate_domain_bounds()

    def _validate_periodicity(self):
        if (not isinstance(self.periodicity, (tuple, list))) or len(self.periodicity) != 3:
            raise ValueError("`periodicity` must be a 3-tuple of bools.")
        if not all(isinstance(periodicity, (bool, numpy.bool_)) for periodicity in self.periodicity):
            raise ValueError("`periodicity` entries must be bools.")

    def _validate_resolution(self):
        if (not isinstance(self.resolution, (tuple, list))) or len(self.resolution) != 3:
            raise ValueError("`resolution` must be a 3-tuple (num_cells_x, num_cells_y, num_cells_z).")
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        if not all(isinstance(num_cells, (int, numpy.integer)) for num_cells in (num_cells_x, num_cells_y, num_cells_z)):
            raise ValueError("`resolution` entries must be ints.")
        if not (num_cells_x > 0 and num_cells_y > 0 and num_cells_z > 0):
            raise ValueError("All entries of `resolution` must be positive.")

    def _validate_domain_bounds(self):
        if (not isinstance(self.domain_bounds, (tuple, list))) or len(self.domain_bounds) != 3:
            raise ValueError("`domain_bounds` must be ((x_min,x_max), (y_min,y_max), (z_min,z_max)).")
        axis_name = ("x", "y", "z")
        for axis_index, bounds in enumerate(self.domain_bounds):
            if (not isinstance(bounds, (tuple, list))) or len(bounds) != 2:
                raise ValueError(f"{axis_name[axis_index]}-axis: bounds must be a 2-tuple (min, max).")
            lo_value, hi_value = bounds
            try:
                lo_float = float(lo_value)
                hi_float = float(hi_value)
            except Exception as error:
                raise ValueError(f"{axis_name[axis_index]}-axis: bounds must be numeric floats.") from error
            if not (numpy.isfinite(lo_float) and numpy.isfinite(hi_float)):
                raise ValueError(f"{axis_name[axis_index]}-axis: bounds must be finite.")
            if not (hi_float > lo_float):
                raise ValueError(f"{axis_name[axis_index]}-axis: max must be > min.")

    @property
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

    @property
    def domain_lengths(
        self,
    ) -> tuple[float, float, float]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        return (
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )


@dataclass(frozen=True)
class ScalarField:
    sim_time: float
    data: numpy.ndarray
    label: str


@dataclass(frozen=True)
class VectorField:
    sim_time: float
    data: numpy.ndarray
    labels: tuple[str, str, str]


## } MODULE
