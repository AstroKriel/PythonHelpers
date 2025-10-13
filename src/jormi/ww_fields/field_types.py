## { MODULE

##
## === DEPENDENCIES
##

import numpy
from dataclasses import dataclass
from functools import cached_property
from jormi.utils import type_utils
from jormi.ww_fields import array_types, array_operators

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
        type_utils.assert_sequence(
            var_obj=self.periodicity,
            var_name="periodicity",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(bool, numpy.bool_),
        )

    def _validate_resolution(self):
        type_utils.assert_sequence(
            var_obj=self.resolution,
            var_name="resolution",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(int, numpy.integer),
        )
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        if not (num_cells_x > 0 and num_cells_y > 0 and num_cells_z > 0):
            raise ValueError("All entries of `resolution` must be positive.")

    def _validate_domain_bounds(self):
        type_utils.assert_sequence(
            var_obj=self.domain_bounds,
            var_name="domain_bounds",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(tuple, list),
        )
        axis_name = ("x", "y", "z")
        for axis_index, bounds in enumerate(self.domain_bounds):
            type_utils.assert_sequence(
                var_obj=bounds,
                var_name=f"domain_bounds[{axis_name[axis_index]}]",
                valid_containers=(tuple, list),
                seq_length=2,
            )
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
        array_types.ensure_sarray(self.data)

    def _validate_label(self):
        type_utils.assert_nonempty_str(
            var_obj=self.label,
            var_name="label",
        )


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
        array_types.ensure_varray(self.data)

    def _validate_labels(self):
        type_utils.assert_sequence(
            var_obj=self.labels,
            var_name="labels",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=str,
        )
        for label_index, label in enumerate(self.labels):
            type_utils.assert_nonempty_str(
                var_obj=label,
                var_name=f"labels[{label_index}]",
            )


@dataclass(frozen=True)
class UnitVectorField(VectorField):
    tol: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        self._validate_unit_magnitude()

    def _validate_unit_magnitude(self):
        q_uvarray = self.data
        q_magn_sq_sarray = array_operators.sum_of_component_squares(q_uvarray)
        if numpy.any(q_magn_sq_sarray == 0):
            raise ValueError("UnitVectorField cannot contain zero vectors.")
        max_error = float(numpy.max(numpy.abs(q_magn_sq_sarray - 1.0)))
        if max_error > self.tol:
            raise ValueError(
                f"Vector magnitude deviates from unit-magnitude=1.0 by max(error)={max_error:.3e} (tol={self.tol}).",
            )

    @classmethod
    def from_vfield(
        cls,
        vfield: VectorField,
        *,
        tol: float = 1e-6,
    ) -> "UnitVectorField":
        return cls(
            data=vfield.data,
            labels=vfield.labels,
            sim_time=vfield.sim_time,
            tol=tol,
        )


def as_unit_vfield(
    vfield: VectorField,
    tol: float = 1e-6,
) -> UnitVectorField:
    ## zero-copy rewrap with validation
    return UnitVectorField.from_vfield(vfield, tol=tol)


##
## === DATA TYPE VALIDATION
##


def ensure_sfield(
    sfield,
) -> None:
    type_utils.assert_type(
        var_obj=sfield,
        valid_types=ScalarField,
    )


def ensure_vfield(
    vfield,
) -> None:
    type_utils.assert_type(
        var_obj=vfield,
        valid_types=VectorField,
    )


def ensure_uvfield(uvfield) -> None:
    type_utils.assert_type(
        var_obj=uvfield,
        valid_types=UnitVectorField,
    )


def ensure_uniform_domain(
    domain_details,
) -> None:
    type_utils.assert_type(
        var_obj=domain_details,
        valid_types=UniformDomain,
    )


def ensure_domain_matches_sfield(
    domain_details: UniformDomain,
    sfield: ScalarField,
) -> None:
    ensure_uniform_domain(domain_details)
    ensure_sfield(sfield)
    if domain_details.resolution != sfield.data.shape:
        raise ValueError(
            f"domain_details resolution {domain_details.resolution} does not match scalar grid {sfield.data.shape}",
        )


def ensure_domain_matches_vfield(
    domain_details: UniformDomain,
    vfield: VectorField,
) -> None:
    ensure_uniform_domain(domain_details)
    ensure_vfield(vfield)  # also accepts subclasses (e.g. UnitVectorField)
    if domain_details.resolution != vfield.data.shape[1:]:
        raise ValueError(
            f"domain_details resolution {domain_details.resolution} does not match vector grid {vfield.data.shape[1:]}",
        )


## } MODULE
