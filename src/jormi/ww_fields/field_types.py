## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass
from functools import cached_property

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
        if not all(isinstance(num_cells, (int, numpy.integer))
                   for num_cells in (num_cells_x, num_cells_y, num_cells_z)):
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


@dataclass(frozen=True)
class ScalarField:
    data: numpy.ndarray
    label: str
    sim_time: float | None = None

    def __post_init__(self):
        self._validate_sim_time()
        self._validate_data()
        self._validate_label()

    def _validate_sim_time(self):
        if self.sim_time is None:
            return
        try:
            sim_time = float(self.sim_time)
        except Exception as error:
            raise ValueError("`sim_time` must be a float.") from error
        if not numpy.isfinite(sim_time):
            raise ValueError("`sim_time` must be finite.")

    def _validate_data(self):
        if not isinstance(self.data, numpy.ndarray):
            raise TypeError("`data` must be a numpy.ndarray.")
        if self.data.ndim != 3:
            raise ValueError(
                f"`data` must have shape (num_cells_x, num_cells_y, num_cells_z); got {self.data.shape}.",
            )

    def _validate_label(self):
        if not isinstance(self.label, str):
            raise TypeError("`label` must be a string.")
        if not self.label:
            raise ValueError("`label` must be a non-empty string.")


@dataclass(frozen=True)
class VectorField:
    data: numpy.ndarray
    labels: tuple[str, str, str]
    sim_time: float | None = None

    def __post_init__(self):
        self._validate_sim_time()
        self._validate_data()
        self._validate_labels()

    def _validate_sim_time(self):
        if self.sim_time is None:
            return
        try:
            sim_time = float(self.sim_time)
        except Exception as error:
            raise ValueError("`sim_time` must be a float.") from error
        if not numpy.isfinite(sim_time):
            raise ValueError("`sim_time` must be finite.")

    def _validate_data(self):
        if not isinstance(self.data, numpy.ndarray):
            raise TypeError("`data` must be a numpy.ndarray.")
        if self.data.ndim != 4:
            raise ValueError(
                f"`data` must have shape (3, num_cells_x, num_cells_y, num_cells_z); got {self.data.shape}.",
            )
        if self.data.shape[0] != 3:
            raise ValueError(
                f"First axis of `data` must have length 3 (x, y, z components); got {self.data.shape[0]}.",
            )

    def _validate_labels(self):
        if (not isinstance(self.labels, (tuple, list))) or len(self.labels) != 3:
            raise ValueError("`labels` must be a 3-tuple of strings.")
        if not all(isinstance(label, str) for label in self.labels):
            raise TypeError("All entries of `labels` must be strings.")
        if not all(label for label in self.labels):
            raise ValueError("All entries of `labels` must be non-empty strings.")


##
## === DATA TYPE VALIDATION
##


def ensure_sarray(
    sarray: numpy.ndarray,
) -> None:
    if sarray.ndim != 3:
        raise ValueError("Scalar field arrays must have shape (num_cells_x, num_cells_y, num_cells_z).")


def ensure_varray(
    varray: numpy.ndarray,
) -> None:
    if (varray.ndim != 4) or (varray.shape[0] != 3):
        raise ValueError("Vector field arrays must have shape (3, num_cells_x, num_cells_y, num_cells_z).")


def ensure_sfield(
    sfield,
) -> None:
    if not isinstance(sfield, ScalarField):
        raise TypeError(f"Expected ScalarField, got {type(sfield).__name__}")


def ensure_vfield(
    vfield,
) -> None:
    if not isinstance(vfield, VectorField):
        raise TypeError(f"Expected VectorField, got {type(vfield).__name__}")


def ensure_uniform_domain(
    domain_details,
) -> None:
    if not isinstance(domain_details, UniformDomain):
        raise TypeError(f"Expected UniformDomain, got {type(domain_details).__name__}")


def ensure_same_grid(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
) -> None:
    if array_a.shape != array_b.shape:
        raise ValueError(f"Grid mismatch: {array_a.shape} vs {array_b.shape}")


def ensure_domain_matches_sfield(
    domain_details: UniformDomain,
    sfield: ScalarField,
) -> None:
    if domain_details.resolution != sfield.data.shape:
        raise ValueError(
            f"domain_details resolution {domain_details.resolution} does not match scalar grid {sfield.data.shape}",
        )


def ensure_domain_matches_vfield(
    domain_details: UniformDomain,
    vfield: VectorField,
) -> None:
    if domain_details.resolution != vfield.data.shape[1:]:
        raise ValueError(
            f"domain_details resolution {domain_details.resolution} does not match vector grid {vfield.data.shape[1:]}",
        )


## } MODULE
