## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from typing import Literal, TypeAlias

from jormi.utils import list_utils
from jormi.ww_types import type_manager

##
## === DATA TYPES
##

IndexValue: TypeAlias = Literal[0, 1, 2]


class CartesianAxis(str, Enum):
    """Default Cartesian axes in 3D with attached index."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def axis_index(
        self,
    ) -> IndexValue:
        """Default index of this axis."""
        if self is CartesianAxis.X:
            return 0
        if self is CartesianAxis.Y:
            return 1
        if self is CartesianAxis.Z:
            return 2
        valid_axes = [cartesian_axis.value for cartesian_axis in CartesianAxis]
        valid_string = list_utils.as_quoted_string(elems=valid_axes)
        raise ValueError(
            f"Unknown CartesianAxis value {self!r}. Expected one of {valid_string}.",
        )


AxisLike: TypeAlias = CartesianAxis | str
AxisTuple: TypeAlias = tuple[CartesianAxis, CartesianAxis, CartesianAxis]

DEFAULT_AXES_ORDER: AxisTuple = (
    CartesianAxis.X,
    CartesianAxis.Y,
    CartesianAxis.Z,
)

##
## === HELPERS FOR AXIS
##


def as_axis(
    *,
    axis: AxisLike,
    param_name: str = "<axis>",
) -> CartesianAxis:
    """Normalise a string ('x', 'y', 'z') or CartesianAxis into a CartesianAxis enum."""
    if isinstance(axis, CartesianAxis):
        return axis
    if isinstance(axis, str):
        axis_lower = axis.lower()
        for candidate_axis in CartesianAxis:
            if candidate_axis.value == axis_lower:
                return candidate_axis
    valid_axes = [cartesian_axis.value for cartesian_axis in CartesianAxis]
    valid_string = list_utils.as_quoted_string(elems=valid_axes)
    raise ValueError(f"`{param_name}` must be one of {valid_string}, got {axis!r}.")


def get_axis_index(
    axis: AxisLike,
) -> IndexValue:
    """Return the default index (0, 1, 2) for a given axis."""
    return as_axis(axis=axis).axis_index


def get_axis_from_index(
    *,
    index: int,
    param_name: str = "<index>",
) -> CartesianAxis:
    """Return the CartesianAxis corresponding to a default index (0, 1, 2)."""
    type_manager.ensure_finite_int(
        param=index,
        param_name=param_name,
        allow_none=False,
        require_positive=False,
    )
    if index == 0:
        return CartesianAxis.X
    if index == 1:
        return CartesianAxis.Y
    if index == 2:
        return CartesianAxis.Z
    valid_indices = [cartesian_axis.axis_index for cartesian_axis in DEFAULT_AXES_ORDER]
    valid_string = list_utils.as_quoted_string(elems=valid_indices)
    raise ValueError(f"`{param_name}` must be one of {valid_string}, got {index!r}.")


##
## === HELPERS FOR AXES
##


def as_axes_tuple(
    *,
    axes: tuple[AxisLike, AxisLike, AxisLike],
    param_name: str = "<axes>",
) -> AxisTuple:
    """
    Convert a length-3 tuple of AxisLike into a tuple of CartesianAxis.

    Validates the outer container shape and then normalises each entry via `as_axis()`.
    """
    type_manager.ensure_sequence(
        param=axes,
        param_name=param_name,
        seq_length=3,
        valid_seq_types=type_manager.RuntimeTypes.Sequences.TupleLike,
    )
    axes_enum_list: list[CartesianAxis] = []
    for axis_index, axis_like in enumerate(axes):
        axis_enum = as_axis(
            axis=axis_like,
            param_name=f"{param_name}[{axis_index}]",
        )
        axes_enum_list.append(axis_enum)
    return (
        axes_enum_list[0],
        axes_enum_list[1],
        axes_enum_list[2],
    )


def ensure_valid_axes_permutation(
    *,
    axes: tuple[AxisLike, AxisLike, AxisLike],
    param_name: str = "<axes>",
) -> None:
    """Ensure `axes` is a permutation of the default axes (order not enforced)."""
    axes_tuple = as_axes_tuple(
        axes=axes,
        param_name=param_name,
    )
    if set(axes_tuple) != set(DEFAULT_AXES_ORDER):
        in_axes = [axis.value for axis in axes_tuple]
        valid_axes = [axis.value for axis in DEFAULT_AXES_ORDER]
        in_string = list_utils.as_quoted_string(elems=in_axes)
        valid_string = list_utils.as_quoted_string(elems=valid_axes)
        raise ValueError(
            f"`{param_name}` must be a permutation of {valid_string}, got {in_string}.",
        )


def ensure_default_axes_order(
    *,
    axes: tuple[AxisLike, AxisLike, AxisLike],
    param_name: str = "<axes>",
) -> None:
    """Ensure `axes` matches the default order exactly: (x, y, z)."""
    axes_tuple = as_axes_tuple(
        axes=axes,
        param_name=param_name,
    )
    ensure_valid_axes_permutation(
        axes=axes_tuple,
        param_name=param_name,
    )
    if axes_tuple != DEFAULT_AXES_ORDER:
        in_axes = [axis.value for axis in axes_tuple]
        valid_axes = [axis.value for axis in DEFAULT_AXES_ORDER]
        in_string = list_utils.as_quoted_string(elems=in_axes)
        valid_string = list_utils.as_quoted_string(elems=valid_axes)
        raise ValueError(
            f"`{param_name}` must be exactly {valid_string}, got {in_string}.",
        )


def is_default_axes_order(
    axes: tuple[AxisLike, AxisLike, AxisLike],
) -> bool:
    """Return True if `axes` corresponds exactly to the default order (x, y, z)."""
    try:
        axes_tuple = as_axes_tuple(
            axes=axes,
            param_name="<axes>",
        )
    except ValueError:
        return False
    return axes_tuple == DEFAULT_AXES_ORDER


## } MODULE
