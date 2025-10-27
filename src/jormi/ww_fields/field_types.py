## { MODULE

##
## === DEPENDENCIES
##

import numpy
from typing import Literal, Self
from functools import cached_property
from dataclasses import dataclass

from jormi.utils import type_utils, array_utils
from jormi.ww_fields import farray_types, farray_operators

##
## === TYPES + CONSTANTS
##

CompAxis = Literal["x", "y", "z"]
CompIndex = Literal[0, 1, 2]

DEFAULT_COMP_AXES_ORDER: tuple[CompAxis, CompAxis, CompAxis] = ("x", "y", "z")
DEFAULT_COMP_AXIS_TO_INDEX: dict[CompAxis, CompIndex] = {"x": 0, "y": 1, "z": 2}


def get_default_comp_index(
    comp_axis: CompAxis,
) -> CompIndex:
    """Map component axis (x,y,z) to index (0,1,2)."""
    try:
        return DEFAULT_COMP_AXIS_TO_INDEX[comp_axis]
    except KeyError as error:
        raise ValueError(f"`comp_axis` must be one of {DEFAULT_COMP_AXES_ORDER}, got {comp_axis}.") from error


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
        type_utils.ensure_sequence(
            var_obj=self.periodicity,
            var_name="periodicity",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(bool, numpy.bool_),
        )

    def _validate_resolution(
        self,
    ) -> None:
        type_utils.ensure_sequence(
            var_obj=self.resolution,
            var_name="resolution",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(int, numpy.integer),
        )
        num_cells_x, num_cells_y, num_cells_z = self.resolution
        if not (num_cells_x > 0 and num_cells_y > 0 and num_cells_z > 0):
            raise ValueError("All entries of `resolution` must be positive.")

    def _validate_domain_bounds(
        self,
    ) -> None:
        type_utils.ensure_sequence(
            var_obj=self.domain_bounds,
            var_name="domain_bounds",
            valid_containers=(tuple, ),
            seq_length=3,
            valid_elem_types=(tuple, list),
        )
        axis_name = ("x", "y", "z")
        for axis_index, bounds in enumerate(self.domain_bounds):
            type_utils.ensure_sequence(
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
    field_label: str
    sim_time: float | None = None

    def __post_init__(self):
        self._validate_sim_time()
        self._validate_data()
        self._validate_label()

    def _validate_sim_time(self):
        type_utils.ensure_finite_float(
            var_obj=self.sim_time,
            var_name="sim_time",
            allow_none=True,
        )

    def _validate_data(self):
        farray_types.ensure_sarray(self.data)

    def _validate_label(self):
        type_utils.ensure_nonempty_str(
            var_obj=self.field_label,
            var_name="field_label",
        )


@dataclass(frozen=True)
class VectorField:
    data: numpy.ndarray
    field_label: str
    comp_labels: tuple[CompAxis, CompAxis, CompAxis] = DEFAULT_COMP_AXES_ORDER
    sim_time: float | None = None

    def __post_init__(
        self,
    ) -> None:
        self._validate_sim_time()
        self._validate_data()
        self._validate_labels()

    def _validate_sim_time(
        self,
    ) -> None:
        type_utils.ensure_finite_float(
            var_obj=self.sim_time,
            var_name="sim_time",
            allow_none=True,
        )

    def _validate_data(
        self,
    ) -> None:
        farray_types.ensure_varray(self.data)

    def _validate_labels(
        self,
    ) -> None:
        type_utils.ensure_nonempty_str(
            var_obj=self.field_label,
            var_name="field_label",
        )
        ensure_default_comp_order(self.comp_labels)

    def get_comp_data(
        self,
        comp_axis: CompAxis,
    ) -> numpy.ndarray:
        """Return a (Nx, Ny, Nz) view of the requested component."""
        comp_index = get_default_comp_index(comp_axis)
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
            comp_labels=vfield.comp_labels,
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
    sfield: ScalarField,
) -> None:
    type_utils.ensure_type(
        var_obj=sfield,
        valid_types=ScalarField,
    )


def ensure_vfield(
    vfield: VectorField,
) -> None:
    type_utils.ensure_type(
        var_obj=vfield,
        valid_types=VectorField,
    )


def ensure_uvfield(
    uvfield: UnitVectorField,
) -> None:
    type_utils.ensure_type(
        var_obj=uvfield,
        valid_types=UnitVectorField,
    )


def ensure_uniform_domain(
    uniform_domain: UniformDomain,
) -> None:
    type_utils.ensure_type(
        var_obj=uniform_domain,
        valid_types=UniformDomain,
    )


def ensure_same_sfield_shape(
    sfield_a: ScalarField,
    sfield_b: ScalarField,
) -> None:
    ensure_sfield(sfield=sfield_a)
    ensure_sfield(sfield=sfield_b)
    array_utils.ensure_same_shape(
        array_a=sfield_a.data,
        array_b=sfield_b.data,
    )


def ensure_same_vfield_shape(
    vfield_a: VectorField,
    vfield_b: VectorField,
) -> None:
    ensure_vfield(vfield=vfield_a)
    ensure_vfield(vfield=vfield_b)
    array_utils.ensure_same_shape(
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
    comp_labels: tuple[CompAxis, CompAxis, CompAxis],
) -> None:
    """Ensure comp_labels is a permutation of DEFAULT_COMP_AXES_ORDER (the order is not enforced)."""
    type_utils.ensure_sequence(
        var_obj=comp_labels,
        var_name="comp_labels",
        valid_containers=(tuple, ),
        seq_length=3,
        valid_elem_types=str,
    )
    for comp_index, comp_label in enumerate(comp_labels):
        type_utils.ensure_nonempty_str(
            var_obj=comp_label,
            var_name=f"comp_labels[{comp_index}]",
        )
        if comp_label not in DEFAULT_COMP_AXES_ORDER:
            raise ValueError(f"`comp_labels` must contain only elements from {DEFAULT_COMP_AXES_ORDER}, got {comp_labels}.")
    if set(comp_labels) != set(DEFAULT_COMP_AXES_ORDER):
        raise ValueError(f"`comp_labels` must be a permutation of {DEFAULT_COMP_AXES_ORDER} (no repeats, none missing), got {comp_labels}.")


def ensure_default_comp_order(
    comp_labels: tuple[CompAxis, CompAxis, CompAxis],
) -> None:
    ensure_all_comp_labels_exist(comp_labels)
    if comp_labels != DEFAULT_COMP_AXES_ORDER:
        raise ValueError(f"`comp_labels` must be exactly {DEFAULT_COMP_AXES_ORDER} in that order, got {comp_labels}.")


## } MODULE
