## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from enum import Enum
from typing import (
    Literal,
    TypeAlias,
    cast,
)

## local
from jormi.ww_validation import validate_enums, validate_types

##
## === DATA TYPES
##

AxisLabel_3D: TypeAlias = Literal["x_0", "x_1", "x_2"]
AxisIndex_3D: TypeAlias = Literal[0, 1, 2]

VALID_3D_AXIS_LABELS: tuple[AxisLabel_3D, AxisLabel_3D, AxisLabel_3D] = (
    "x_0",
    "x_1",
    "x_2",
)
VALID_3D_AXIS_INDICES: tuple[AxisIndex_3D, AxisIndex_3D, AxisIndex_3D] = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class AxisParams:
    """Label and integer index pair for a single Cartesian axis."""

    axis_label: str
    axis_index: int


class CartesianAxis_3D(str, Enum):
    """Default Cartesian axes in 3D with an explicit label and axis_index."""

    _axis_label: AxisLabel_3D
    _axis_index: AxisIndex_3D

    X0 = AxisParams(
        axis_label=VALID_3D_AXIS_LABELS[0],
        axis_index=VALID_3D_AXIS_INDICES[0],
    )
    X1 = AxisParams(
        axis_label=VALID_3D_AXIS_LABELS[1],
        axis_index=VALID_3D_AXIS_INDICES[1],
    )
    X2 = AxisParams(
        axis_label=VALID_3D_AXIS_LABELS[2],
        axis_index=VALID_3D_AXIS_INDICES[2],
    )

    def __new__(
        cls,
        axis_params: AxisParams,
    ) -> "CartesianAxis_3D":
        validate_types.ensure_type(
            param=axis_params.axis_label,
            valid_types=validate_types.RuntimeTypes.Strings.StringLike,
            param_name="axis_label",
        )
        if axis_params.axis_label not in VALID_3D_AXIS_LABELS:
            raise ValueError(f"`axis_label` is invalid: {axis_params.axis_label!r}.")
        validate_types.ensure_type(
            param=axis_params.axis_index,
            valid_types=validate_types.RuntimeTypes.Numerics.IntLike,
            param_name="axis_index",
        )
        if axis_params.axis_index not in VALID_3D_AXIS_INDICES:
            raise ValueError(f"`axis_index` is invalid: {axis_params.axis_index!r}.")
        obj = str.__new__(cls, axis_params.axis_label)
        obj._value_ = axis_params.axis_label
        obj._axis_label = cast(AxisLabel_3D, axis_params.axis_label)  # pyright: ignore[reportUnnecessaryCast]
        obj._axis_index = cast(AxisIndex_3D, axis_params.axis_index)  # pyright: ignore[reportUnnecessaryCast]
        return obj

    @property
    def axis_label(
        self,
    ) -> AxisLabel_3D:
        return self._axis_label

    @property
    def axis_index(
        self,
    ) -> AxisIndex_3D:
        return self._axis_index


AxisLike_3D: TypeAlias = CartesianAxis_3D | str | int
AxisTuple_3D: TypeAlias = tuple[CartesianAxis_3D, CartesianAxis_3D, CartesianAxis_3D]

DEFAULT_3D_AXES_ORDER: AxisTuple_3D = (
    CartesianAxis_3D.X0,
    CartesianAxis_3D.X1,
    CartesianAxis_3D.X2,
)


def as_axis(
    axis: AxisLike_3D,
    *,
    param_name: str = "<axis>",
) -> CartesianAxis_3D:
    """
    Resolve `axis` into a canonical `CartesianAxis_3D` member.

    Accepts:
      - A `CartesianAxis_3D` member (e.g. `CartesianAxis_3D.X1`)
      - An integer axis index: 0, 1, 2
      - A string axis identifier (case-insensitive), matching either:
          * the axis label: "x_0", "x_1", "x_2"
          * the Enum member name: "X0", "X1", "X2"
    """
    validate_types.ensure_type(
        param=axis,
        valid_types=(int, str, CartesianAxis_3D),
        param_name=param_name,
    )
    if isinstance(axis, CartesianAxis_3D):
        return axis
    if isinstance(axis, int):
        if axis == VALID_3D_AXIS_INDICES[0]:
            return CartesianAxis_3D.X0
        if axis == VALID_3D_AXIS_INDICES[1]:
            return CartesianAxis_3D.X1
        if axis == VALID_3D_AXIS_INDICES[2]:
            return CartesianAxis_3D.X2
        raise ValueError(
            f"`{param_name}` must be one of {VALID_3D_AXIS_INDICES}, got {axis!r}.",
        )
    return cast(
        CartesianAxis_3D,
        validate_enums.resolve_member(
            member=axis,
            valid_enums=CartesianAxis_3D,
        ),
    )


def get_axis_label(
    axis: AxisLike_3D,
) -> AxisLabel_3D:
    return as_axis(axis=axis).axis_label


def get_axis_index(
    axis: AxisLike_3D,
) -> AxisIndex_3D:
    return as_axis(axis=axis).axis_index


## } MODULE
