## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from enum import Enum
from typing import TypeAlias, cast

## local
from jormi.ww_types import check_enums

##
## === ENUM DEFINITIONS
##


class Center(str, Enum):

    Center = "center"


class QuadrantCorner(str, Enum):
    """Corner placements using mpl-loc strings."""

    TopLeft = "upper left"
    TopRight = "upper right"
    BottomLeft = "lower left"
    BottomRight = "lower right"


class QuadrantEdge(str, Enum):
    """Edge-center placements using mpl-loc strings."""

    Top = "upper center"
    Left = "center left"
    Right = "center right"
    Bottom = "lower center"


class Side(str, Enum):

    Top = "top"
    Left = "left"
    Right = "right"
    Bottom = "bottom"


##
## === TYPE HINTS + RUNTIME TYPES
## TypeHints defines categories for annotations.
## RuntimeTypes converts them into tuples for enum_checks validation.
##


class TypeHints:
    """Type-hint groupings for box position and Matplotlib placement parameters."""

    PositionLike: TypeAlias = check_enums.EnumMemberLike

    class Box:

        Center = Center
        Corner = QuadrantCorner
        Edge = QuadrantEdge
        Side = Side

    class MPL:
        """Type-hint groupings for mpl-style parameters."""

        AnchorLike = QuadrantCorner | QuadrantEdge | Center
        AlignLike = Side | Center


class RuntimeTypes:
    """Runtime enum tuples derived from TypeHints for isinstance-based position checks."""

    class Box:

        Center = check_enums.as_runtime_type(TypeHints.Box.Center)
        Corner = check_enums.as_runtime_type(TypeHints.Box.Corner)
        Edge = check_enums.as_runtime_type(TypeHints.Box.Edge)
        Side = check_enums.as_runtime_type(TypeHints.Box.Side)

    class MPL:
        """Runtime enum tuples for mpl-style rules."""

        AnchorLike = check_enums.as_runtime_type(TypeHints.MPL.AnchorLike)
        AlignLike = check_enums.as_runtime_type(TypeHints.MPL.AlignLike)


##
## === BOX RULES
##


def ensure_box_corner(
    corner: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
        param_name=param_name,
    )


def ensure_box_edge(
    edge: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
        param_name=param_name,
    )


def ensure_box_center(
    center: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
        param_name=param_name,
    )


def ensure_box_side(
    side: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
        param_name=param_name,
    )


def as_box_corner(
    corner: TypeHints.PositionLike,
) -> TypeHints.Box.Corner:
    ensure_box_corner(
        corner=corner,
        param_name="corner",
    )
    resolved_corner = check_enums.resolve_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
    )
    return cast(TypeHints.Box.Corner, resolved_corner)


def as_box_edge(
    edge: TypeHints.PositionLike,
) -> TypeHints.Box.Edge:
    ensure_box_edge(
        edge=edge,
        param_name="edge",
    )
    resolved_edge = check_enums.resolve_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
    )
    return cast(TypeHints.Box.Edge, resolved_edge)


def as_box_center(
    center: TypeHints.PositionLike,
) -> TypeHints.Box.Center:
    ensure_box_center(
        center=center,
        param_name="center",
    )
    resolved_center = check_enums.resolve_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
    )
    return cast(TypeHints.Box.Center, resolved_center)


def as_box_side(
    side: TypeHints.PositionLike,
) -> TypeHints.Box.Side:
    ensure_box_side(
        side=side,
        param_name="side",
    )
    resolved_side = check_enums.resolve_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
    )
    return cast(TypeHints.Box.Side, resolved_side)


##
## === MATPLOTLIB RULES
##


def ensure_mpl_anchor(
    position: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `position` is valid for mpl-loc placement."""
    check_enums.ensure_valid_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
        param_name=param_name,
    )


def ensure_mpl_ha(
    ha: TypeHints.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `ha` is valid for mpl-ha."""
    check_enums.ensure_member_in(
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
    """Ensure `va` is valid for mpl-va."""
    check_enums.ensure_member_in(
        member=va,
        valid_members=(
            TypeHints.Box.Side.Top,
            TypeHints.Box.Side.Bottom,
            TypeHints.Box.Center.Center,
        ),
        param_name=param_name,
    )


def as_mpl_anchor(
    position: TypeHints.PositionLike,
) -> TypeHints.MPL.AnchorLike:
    """Resolve `position` to an anchor enum member for mpl-loc placement."""
    ensure_mpl_anchor(
        position=position,
        param_name="position",
    )
    resolved_position = check_enums.resolve_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
    )
    return cast(TypeHints.MPL.AnchorLike, resolved_position)


def as_mpl_ha(
    ha: TypeHints.PositionLike,
) -> TypeHints.MPL.AlignLike:
    """Resolve `ha` to an enum member valid for mpl-ha."""
    ensure_mpl_ha(
        ha=ha,
        param_name="ha",
    )
    ## the previous check already ensured the correct set
    resolved_ha = check_enums.resolve_member(
        member=ha,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return cast(TypeHints.MPL.AlignLike, resolved_ha)


def as_mpl_va(
    va: TypeHints.PositionLike,
) -> TypeHints.MPL.AlignLike:
    """Resolve `va` to an enum member valid for mpl-va."""
    ensure_mpl_va(
        va=va,
        param_name="va",
    )
    ## the previous check already ensured the correct set
    resolved_va = check_enums.resolve_member(
        member=va,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return cast(TypeHints.MPL.AlignLike, resolved_va)


## } MODULE