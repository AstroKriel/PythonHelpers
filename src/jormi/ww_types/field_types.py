## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Literal, Self
from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import type_manager, array_types, farray_types
from jormi.ww_fields import farray_operators

##
## === DATA TYPES
##

AxisName = Literal["x", "y", "z"]
IndexNames = Literal["i", "j", "k"]
IndexValue = Literal[0, 1, 2]

AXES_NAMES: tuple[AxisName, AxisName, AxisName] = ("x", "y", "z")
INDEX_NAMES: tuple[IndexNames, IndexNames, IndexNames] = ("i", "j", "k")
AXIS_NAME_TO_INDEX_VALUE: dict[AxisName, IndexValue] = {"x": 0, "y": 1, "z": 2}


def get_comp_index(
    comp_axis: AxisName,
) -> IndexValue:
    """Map component axis (x,y,z) to index (0,1,2)."""
    try:
        return AXIS_NAME_TO_INDEX_VALUE[comp_axis]
    except KeyError as error:
        raise ValueError(f"`comp_axis` must be one of {AXES_NAMES}, got {comp_axis}.") from error


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
            param_name="periodicity",
            seq_length=3,
            valid_seq_types=(tuple, ),
            valid_elem_types=(bool, numpy.bool_),
        )

    def _validate_resolution(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.resolution,
            param_name="resolution",
            seq_length=3,
            valid_seq_types=(tuple, ),
            valid_elem_types=(int, numpy.integer),
        )
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        if (num_cells_x <= 0) or (num_cells_y <= 0) or (num_cells_z <= 0):
            raise ValueError("All `resolution` entries must be positive.")

    def _validate_domain_bounds(
        self,
    ) -> None:
        type_manager.ensure_sequence(
            param=self.domain_bounds,
            param_name="domain_bounds",
            seq_length=3,
            valid_seq_types=(tuple, ),
            valid_elem_types=(tuple, list),
        )
        for axis_index, bounds in enumerate(self.domain_bounds):
            type_manager.ensure_sequence(
                param=bounds,
                param_name=f"domain_bounds[{AXES_NAMES[axis_index]}]",
                seq_length=2,
                valid_seq_types=(tuple, list),
            )
            lo_value, hi_value = bounds
            try:
                lo_float = float(lo_value)
                hi_float = float(hi_value)
            except Exception as error:
                raise ValueError(f"{AXES_NAMES[axis_index]}-axis: bounds must be numeric floats.") from error
            if not (numpy.isfinite(lo_float) and numpy.isfinite(hi_float)):
                raise ValueError(f"{AXES_NAMES[axis_index]}-axis: bounds must be finite.")
            if not (hi_float > lo_float):
                raise ValueError(f"{AXES_NAMES[axis_index]}-axis: max must be > min.")

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
    field_label: str
    sim_time: float | None = None

    def __post_init__(self):
        self._validate_sim_time()
        self._validate_data()
        self._validate_label()

    def _validate_sim_time(self):
        type_manager.ensure_finite_float(
            param=self.sim_time,
            param_name="sim_time",
            allow_none=True,
        )

    def _validate_data(self):
        farray_types.ensure_sarray(self.data)

    def _validate_label(self):
        type_manager.ensure_nonempty_str(
            param=self.field_label,
            param_name="field_label",
        )


@dataclass(frozen=True)
class VectorField:
    data: numpy.ndarray
    field_label: str
    comp_axes: tuple[AxisName, AxisName, AxisName] = AXES_NAMES
    sim_time: float | None = None

    def __post_init__(
        self,
    ) -> None:
        self._validate_sim_time()
        self._validate_data()
        self._validate_axes()

    def _validate_sim_time(
        self,
    ) -> None:
        type_manager.ensure_finite_float(
            param=self.sim_time,
            param_name="sim_time",
            allow_none=True,
        )

    def _validate_data(
        self,
    ) -> None:
        farray_types.ensure_varray(self.data)

    def _validate_axes(
        self,
    ) -> None:
        type_manager.ensure_nonempty_str(
            param=self.field_label,
            param_name="field_label",
        )
        ensure_default_comp_order(self.comp_axes)

    def get_comp_data(
        self,
        comp_axis: AxisName,
    ) -> numpy.ndarray:
        """Return a (Nx, Ny, Nz) view of the requested component."""
        comp_index = get_comp_index(comp_axis)
        return self.data[comp_index, ...]


@dataclass(frozen=True)
class UnitVectorField(VectorField):
    tol: float = 1e-6

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()
        self._validate_unit_magnitude()

    def _validate_unit_magnitude(
        self,
    ) -> None:
        q_uvarray = self.data
        q_magn_sq_sarray = farray_operators.sum_of_squared_components(q_uvarray)
        if not numpy.all(numpy.isfinite(q_magn_sq_sarray)):
            raise ValueError("UnitVectorField should not contain any NaN/Inf magnitudes.")
        if numpy.any(q_magn_sq_sarray <= 1e-300):
            raise ValueError("UnitVectorField should not contain any (near-)zero vectors.")
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
    ) -> Self:
        return cls(
            data=vfield.data,
            field_label=vfield.field_label,
            comp_axes=vfield.comp_axes,
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
## === TYPE VALIDATION
##


def ensure_sfield(
    sfield: ScalarField,
) -> None:
    type_manager.ensure_type(
        param=sfield,
        valid_types=ScalarField,
    )


def ensure_vfield(
    vfield: VectorField,
) -> None:
    type_manager.ensure_type(
        param=vfield,
        valid_types=VectorField,
    )


def ensure_uvfield(
    uvfield: UnitVectorField,
) -> None:
    type_manager.ensure_type(
        param=uvfield,
        valid_types=UnitVectorField,
    )


def ensure_uniform_domain(
    uniform_domain: UniformDomain,
) -> None:
    type_manager.ensure_type(
        param=uniform_domain,
        valid_types=UniformDomain,
    )


def ensure_same_sfield_shape(
    sfield_a: ScalarField,
    sfield_b: ScalarField,
) -> None:
    ensure_sfield(sfield=sfield_a)
    ensure_sfield(sfield=sfield_b)
    array_types.ensure_same_shape(
        array_a=sfield_a.data,
        array_b=sfield_b.data,
    )


def ensure_same_vfield_shape(
    vfield_a: VectorField,
    vfield_b: VectorField,
) -> None:
    ensure_vfield(vfield=vfield_a)
    ensure_vfield(vfield=vfield_b)
    array_types.ensure_same_shape(
        array_a=vfield_a.data,
        array_b=vfield_b.data,
    )


def ensure_domain_matches_sfield(
    uniform_domain: UniformDomain,
    sfield: ScalarField,
) -> None:
    ensure_uniform_domain(uniform_domain)
    ensure_sfield(sfield)
    if uniform_domain.resolution != sfield.data.shape:
        raise ValueError(
            f"Domain resolution {uniform_domain.resolution} does not match scalar grid {sfield.data.shape}",
        )


def ensure_domain_matches_vfield(
    uniform_domain: UniformDomain,
    vfield: VectorField,
) -> None:
    ensure_uniform_domain(uniform_domain)
    ensure_vfield(vfield)  # also accepts subclasses (e.g. UnitVectorField)
    if uniform_domain.resolution != vfield.data.shape[1:]:
        raise ValueError(
            f"Domain resolution {uniform_domain.resolution} does not match vector grid {vfield.data.shape[1:]}",
        )


def ensure_all_comp_labels_exist(
    comp_axes: tuple[AxisName, AxisName, AxisName],
) -> None:
    """Ensure comp_axes is a permutation of AXES_NAMES (the order is not enforced)."""
    type_manager.ensure_sequence(
        param=comp_axes,
        param_name="comp_axes",
        seq_length=3,
        valid_seq_types=(tuple, ),
        valid_elem_types=str,
    )
    for comp_index, comp_label in enumerate(comp_axes):
        type_manager.ensure_nonempty_str(
            param=comp_label,
            param_name=f"comp_axes[{comp_index}]",
        )
        if comp_label not in AXES_NAMES:
            raise ValueError(f"`comp_axes` must contain only elements from {AXES_NAMES}, got {comp_axes}.")
    if set(comp_axes) != set(AXES_NAMES):
        raise ValueError(
            f"`comp_axes` must be a permutation of {AXES_NAMES} (no repeats, none missing), got {comp_axes}.",
        )


def ensure_default_comp_order(
    comp_axes: tuple[AxisName, AxisName, AxisName],
) -> None:
    ensure_all_comp_labels_exist(comp_axes)
    if comp_axes != AXES_NAMES:
        raise ValueError(f"`comp_axes` must be exactly {AXES_NAMES} in that order, got {comp_axes}.")


## } MODULE
