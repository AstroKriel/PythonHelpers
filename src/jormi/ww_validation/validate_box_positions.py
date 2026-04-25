## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import cast

## local
from jormi.ww_validation import validate_enums
from jormi.ww_types import box_positions
from jormi.ww_types.box_positions import _AnchorLike, _AlignLike

##
## === RUNTIME TYPES
## derived from `box_positions.Positions` for isinstance-based validation inside `ensure_*`/`as_*`.
##


class RuntimeTypes:
    class Box:
        Center = validate_enums.as_runtime_type(box_positions.Positions.Center)
        Corner = validate_enums.as_runtime_type(box_positions.Positions.Corner)
        Edge = validate_enums.as_runtime_type(box_positions.Positions.Edge)
        Side = validate_enums.as_runtime_type(box_positions.Positions.Side)

    class MPL:
        AnchorLike = validate_enums.as_runtime_type(box_positions._AnchorLike)
        AlignLike = validate_enums.as_runtime_type(box_positions._AlignLike)


##
## === BOX RULES
##


def ensure_box_corner(
    corner: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    validate_enums.ensure_valid_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
        param_name=param_name,
    )


def ensure_box_edge(
    edge: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    validate_enums.ensure_valid_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
        param_name=param_name,
    )


def ensure_box_center(
    center: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    validate_enums.ensure_valid_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
        param_name=param_name,
    )


def ensure_box_side(
    side: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    validate_enums.ensure_valid_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
        param_name=param_name,
    )


def as_box_corner(
    corner: box_positions.Positions.PositionLike,
) -> box_positions.Positions.Corner:
    ensure_box_corner(
        corner=corner,
        param_name="corner",
    )
    resolved_corner = validate_enums.resolve_member(
        member=corner,
        valid_enums=RuntimeTypes.Box.Corner,
    )
    return cast(box_positions.Positions.Corner, resolved_corner)


def as_box_edge(
    edge: box_positions.Positions.PositionLike,
) -> box_positions.Positions.Edge:
    ensure_box_edge(
        edge=edge,
        param_name="edge",
    )
    resolved_edge = validate_enums.resolve_member(
        member=edge,
        valid_enums=RuntimeTypes.Box.Edge,
    )
    return cast(box_positions.Positions.Edge, resolved_edge)


def as_box_center(
    center: box_positions.Positions.PositionLike,
) -> box_positions.Positions.Center:
    ensure_box_center(
        center=center,
        param_name="center",
    )
    resolved_center = validate_enums.resolve_member(
        member=center,
        valid_enums=RuntimeTypes.Box.Center,
    )
    return cast(box_positions.Positions.Center, resolved_center)


def as_box_side(
    side: box_positions.Positions.PositionLike,
) -> box_positions.Positions.Side:
    ensure_box_side(
        side=side,
        param_name="side",
    )
    resolved_side = validate_enums.resolve_member(
        member=side,
        valid_enums=RuntimeTypes.Box.Side,
    )
    return cast(box_positions.Positions.Side, resolved_side)


##
## === MATPLOTLIB RULES
##


def ensure_mpl_anchor(
    position: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `position` is valid for mpl-loc placement."""
    validate_enums.ensure_valid_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
        param_name=param_name,
    )


def ensure_mpl_ha(
    ha: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `ha` is valid for mpl-ha."""
    validate_enums.ensure_member_in(
        member=ha,
        valid_members=(
            box_positions.Positions.Side.Left,
            box_positions.Positions.Side.Right,
            box_positions.Positions.Center.Center,
        ),
        param_name=param_name,
    )


def ensure_mpl_va(
    va: box_positions.Positions.PositionLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `va` is valid for mpl-va."""
    validate_enums.ensure_member_in(
        member=va,
        valid_members=(
            box_positions.Positions.Side.Top,
            box_positions.Positions.Side.Bottom,
            box_positions.Positions.Center.Center,
        ),
        param_name=param_name,
    )


def as_mpl_anchor(
    position: box_positions.Positions.PositionLike,
) -> _AnchorLike:
    """Resolve `position` to an anchor enum member for mpl-loc placement."""
    ensure_mpl_anchor(
        position=position,
        param_name="position",
    )
    resolved_position = validate_enums.resolve_member(
        member=position,
        valid_enums=RuntimeTypes.MPL.AnchorLike,
    )
    return cast(_AnchorLike, resolved_position)


def as_mpl_ha(
    ha: box_positions.Positions.PositionLike,
) -> _AlignLike:
    """Resolve `ha` to an enum member valid for mpl-ha."""
    ensure_mpl_ha(
        ha=ha,
        param_name="ha",
    )
    ## the previous check already ensured the correct set
    resolved_ha = validate_enums.resolve_member(
        member=ha,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return cast(_AlignLike, resolved_ha)


def as_mpl_va(
    va: box_positions.Positions.PositionLike,
) -> _AlignLike:
    """Resolve `va` to an enum member valid for mpl-va."""
    ensure_mpl_va(
        va=va,
        param_name="va",
    )
    ## the previous check already ensured the correct set
    resolved_va = validate_enums.resolve_member(
        member=va,
        valid_enums=RuntimeTypes.MPL.AlignLike,
    )
    return cast(_AlignLike, resolved_va)


## } MODULE
