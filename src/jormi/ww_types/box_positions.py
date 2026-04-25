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
## === POSITIONS
##


class Positions:
    """
    Public namespace for box position enums; use for both type annotations and value selection.

    For mpl-specific usage, prefer `MPLPositions`, which pre-filters members to what each mpl
    parameter actually accepts.
    """

    PositionLike: TypeAlias = check_enums.EnumMemberLike

    class Center(str, Enum):
        """The single centred position. Valid for mpl `ha`/`va` and mpl `loc`."""

        Center = "center"

    class Corner(str, Enum):
        """The four corners of a box. Valid for mpl `loc` anchor and quadrant mask selection."""

        TopLeft = "upper left"
        TopRight = "upper right"
        BottomLeft = "lower left"
        BottomRight = "lower right"

    class Edge(str, Enum):
        """The four edge midpoints of a box. Valid for mpl `loc` anchor and colorbar placement."""

        Top = "upper center"
        Left = "center left"
        Right = "center right"
        Bottom = "lower center"

    class Side(str, Enum):
        """The four faces of a box. Valid for colorbar side and axis label side; not for mpl `loc`."""

        Top = "top"
        Left = "left"
        Right = "right"
        Bottom = "bottom"


##
## === MPL POSITIONS
##


class MPLPositions:
    """
    Filtered views of `Positions` organised by mpl parameter concept.

    Use instead of `Positions` when selecting a value for a specific mpl parameter; the
    sub-class pre-filters members to what that parameter actually accepts, so no knowledge
    of the underlying geometry types is required.
    """

    class Anchor:
        """
        Valid positions for mpl `loc` (legend anchor, colorbar anchor).

        Accepts corners, edge midpoints, and center; not sides.
        """

        Corner = Positions.Corner
        Edge = Positions.Edge
        Center = Positions.Center

    class Align:
        """
        Valid positions for mpl `ha` and `va` (text horizontal and vertical alignment).

        Accepts sides and center; not corners or edges.
        """

        Side = Positions.Side
        Center = Positions.Center


##
## === TYPE ALIASES
## mpl-specific subsets; private because callers select from Positions.* members.
##

_AnchorLike: TypeAlias = Positions.Corner | Positions.Edge | Positions.Center
_AlignLike: TypeAlias = Positions.Side | Positions.Center

##
## === RUNTIME TYPES
## Derived from Positions for isinstance-based validation inside ensure_*/as_* functions.
##


class RuntimeTypes:

    class Box:

        Center = check_enums.as_runtime_type(Positions.Center)
        Corner = check_enums.as_runtime_type(Positions.Corner)
        Edge = check_enums.as_runtime_type(Positions.Edge)
        Side = check_enums.as_runtime_type(Positions.Side)

    class MPL:

        AnchorLike = check_enums.as_runtime_type(_AnchorLike)
        AlignLike = check_enums.as_runtime_type(_AlignLike)


##
## === BOX RULES
##


def ensure_box_corner(
    corner: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
        param_name=param_name,
    )


def ensure_box_edge(
    edge: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
        param_name=param_name,
    )


def ensure_box_center(
    center: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
        param_name=param_name,
    )


def ensure_box_side(
    side: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    check_enums.ensure_valid_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
        param_name=param_name,
    )


def as_box_corner(
    corner: Positions.PositionLike,
) -> Positions.Corner:
    ensure_box_corner(
        corner=corner,
        param_name="corner",
    )
    resolved_corner = check_enums.resolve_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
    )
    return cast(Positions.Corner, resolved_corner)


def as_box_edge(
    edge: Positions.PositionLike,
) -> Positions.Edge:
    ensure_box_edge(
        edge=edge,
        param_name="edge",
    )
    resolved_edge = check_enums.resolve_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
    )
    return cast(Positions.Edge, resolved_edge)


def as_box_center(
    center: Positions.PositionLike,
) -> Positions.Center:
    ensure_box_center(
        center=center,
        param_name="center",
    )
    resolved_center = check_enums.resolve_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
    )
    return cast(Positions.Center, resolved_center)


def as_box_side(
    side: Positions.PositionLike,
) -> Positions.Side:
    ensure_box_side(
        side=side,
        param_name="side",
    )
    resolved_side = check_enums.resolve_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
    )
    return cast(Positions.Side, resolved_side)


##
## === MATPLOTLIB RULES
##


def ensure_mpl_anchor(
    position: Positions.PositionLike,
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
    ha: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `ha` is valid for mpl-ha."""
    check_enums.ensure_member_in(
        member=ha,
        valid_members=(
            Positions.Side.Left,
            Positions.Side.Right,
            Positions.Center.Center,
        ),
        param_name=param_name,
    )


def ensure_mpl_va(
    va: Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `va` is valid for mpl-va."""
    check_enums.ensure_member_in(
        member=va,
        valid_members=(
            Positions.Side.Top,
            Positions.Side.Bottom,
            Positions.Center.Center,
        ),
        param_name=param_name,
    )


def as_mpl_anchor(
    position: Positions.PositionLike,
) -> _AnchorLike:
    """Resolve `position` to an anchor enum member for mpl-loc placement."""
    ensure_mpl_anchor(
        position=position,
        param_name="position",
    )
    resolved_position = check_enums.resolve_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
    )
    return cast(_AnchorLike, resolved_position)


def as_mpl_ha(
    ha: Positions.PositionLike,
) -> _AlignLike:
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
    return cast(_AlignLike, resolved_ha)


def as_mpl_va(
    va: Positions.PositionLike,
) -> _AlignLike:
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
    return cast(_AlignLike, resolved_va)


## } MODULE
