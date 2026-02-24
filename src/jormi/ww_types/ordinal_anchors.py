## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from typing import TypeAlias

from jormi.ww_types import enum_checks

##
## === ENUM DEFINITIONS
##


class _Center(str, Enum):

    Center = "center"


class _QuadrantCorner(str, Enum):
    """
    Quadrant corner positions.

    Values match Matplotlib's `loc` strings.
    """

    TopLeft = "upper left"
    TopRight = "upper right"
    BottomLeft = "lower left"
    BottomRight = "lower right"


class _QuadrantEdge(str, Enum):
    """
    Quadrant edge-centre positions.

    Values match Matplotlib's `loc` strings.
    """

    Top = "upper center"
    Left = "center left"
    Right = "center right"
    Bottom = "lower center"


class _Side(str, Enum):

    Top = "top"
    Left = "left"
    Right = "right"
    Bottom = "bottom"


##
## === TYPE HINTS + RUNTIME TYPES
##


class TypeHints:

    PositionLike: TypeAlias = enum_checks.EnumMemberLike

    class Box:
        Center = _Center
        Corner = _QuadrantCorner
        Edge = _QuadrantEdge
        Side = _Side

    class MPL:
        AnchorLike = _QuadrantCorner | _QuadrantEdge | _Center
        AlignLike = _Side | _Center


class RuntimeTypes:

    class Box:
        Center = enum_checks.as_runtime_type(TypeHints.Box.Center)
        Corner = enum_checks.as_runtime_type(TypeHints.Box.Corner)
        Edge = enum_checks.as_runtime_type(TypeHints.Box.Edge)
        Side = enum_checks.as_runtime_type(TypeHints.Box.Side)

    class MPL:
        AnchorLike = enum_checks.as_runtime_type(TypeHints.MPL.AnchorLike)
        AlignLike = enum_checks.as_runtime_type(TypeHints.MPL.AlignLike)


##
## === BASIS RULES
##


def ensure_box_corner(
    corner: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    enum_checks.ensure_valid_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
        param_name=param_name,
    )


def ensure_box_edge(
    edge: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    enum_checks.ensure_valid_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
        param_name=param_name,
    )


def ensure_box_center(
    center: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    enum_checks.ensure_valid_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
        param_name=param_name,
    )


def ensure_box_side(
    side: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    enum_checks.ensure_valid_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
        param_name=param_name,
    )


def as_box_corner(
    corner: TypeHints.PositionLike,
) -> TypeHints.Box.Corner:
    resolved_corner = enum_checks.resolve_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
    )
    return resolved_corner  # type: ignore[return-value]


def as_box_edge(
    edge: TypeHints.PositionLike,
) -> TypeHints.Box.Edge:
    resolved_edge = enum_checks.resolve_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
    )
    return resolved_edge  # type: ignore[return-value]


def as_box_center(
    center: TypeHints.PositionLike,
) -> TypeHints.Box.Center:
    resolved_center = enum_checks.resolve_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
    )
    return resolved_center  # type: ignore[return-value]


def as_box_side(
    side: TypeHints.PositionLike,
) -> TypeHints.Box.Side:
    resolved_side = enum_checks.resolve_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
    )
    return resolved_side  # type: ignore[return-value]


##
## === MATPLOTLIB RULES
##


def ensure_mpl_anchor(
    position: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `position` is valid for Matplotlib `loc`."""
    enum_checks.ensure_valid_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
        param_name=param_name,
    )


def as_mpl_anchor(
    position: TypeHints.PositionLike,
) -> TypeHints.MPL.AnchorLike:
    """Resolve `position` into a `loc`-compatible enum member."""
    resolved_position = enum_checks.resolve_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
    )
    return resolved_position  # type: ignore[return-value]


def ensure_mpl_ha(
    ha: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """
    Ensure `ha` is valid for Matplotlib `horizontalalignment`.

    Allowed: Left, Right, Center.
    """
    enum_checks.ensure_member_in(
        member=ha,
        valid_members=(
            TypeHints.Box.Side.Left,
            TypeHints.Box.Side.Right,
            TypeHints.Box.Center.Center,
        ),
        param_name=param_name,
    )


def ensure_mpl_va(
    va: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """
    Ensure `va` is valid for Matplotlib `verticalalignment`.

    Allowed: Top, Bottom, Center.

    Note: Matplotlib also supports baseline alignments; these are intentionally excluded
    by this basis.
    """
    enum_checks.ensure_member_in(
        member=va,
        valid_members=(
            TypeHints.Box.Side.Top,
            TypeHints.Box.Side.Bottom,
            TypeHints.Box.Center.Center,
        ),
        param_name=param_name,
    )


def as_mpl_ha(
    ha: TypeHints.PositionLike,
) -> TypeHints.MPL.AlignLike:
    """Resolve `ha` into a basis enum member valid for Matplotlib `ha`."""
    ensure_mpl_ha(
        ha=ha,
        param_name="ha",
    )
    resolved_ha = enum_checks.resolve_member(
        member=ha,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return resolved_ha # type: ignore[return-value]


def as_mpl_va(
    va: TypeHints.PositionLike,
) -> TypeHints.MPL.AlignLike:
    """Resolve `va` into a basis enum member valid for Matplotlib `va`."""
    ensure_mpl_va(
        va=va,
        param_name="va",
    )
    resolved_va = enum_checks.resolve_member(
        member=va,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return resolved_va  # type: ignore[return-value]


## } MODULE
