## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from typing import TypeAlias

from jormi.utils import list_utils

##
## === TYPE DEFINITIONS
##


class VerticalAnchor(str, Enum):
    """Discrete vertical anchors within a box/layout."""

    Top = "top"
    Center = "center"
    Bottom = "bottom"


class HorizontalAnchor(str, Enum):
    """Discrete horizontal anchors within a box/layout."""

    Left = "left"
    Center = "center"
    Right = "right"


VerticalAnchorLike: TypeAlias = VerticalAnchor | str
HorizontalAnchorLike: TypeAlias = HorizontalAnchor | str
AnchorLike: TypeAlias = VerticalAnchorLike | HorizontalAnchorLike

##
## === TYPE CONVERTERS
##


def as_vertical_anchor(
    anchor: VerticalAnchorLike,
) -> VerticalAnchor:
    """
    Convert into a canonical VerticalAnchor.

    Accepts either:
    - Enum members: VerticalAnchor.Top / Center / Bottom
    - Strings matching the Enum value (case insensitive)
    - Strings matching the Enum name (case insensitive)

    Returns one of:
    - VerticalAnchor.Top
    - VerticalAnchor.Center
    - VerticalAnchor.Bottom
    """
    if isinstance(anchor, VerticalAnchor):
        return anchor
    if not isinstance(anchor, str):
        raise TypeError(
            f"Expected a VerticalAnchor or str; got {type(anchor).__name__}.",
        )
    anchor_lower = anchor.lower()
    resolved_anchor = None
    for vertical_anchor in VerticalAnchor:
        is_like_anchor_value = (anchor_lower == vertical_anchor.value.lower())
        is_like_anchor_name = (anchor_lower == vertical_anchor.name.lower())
        if is_like_anchor_value or is_like_anchor_name:
            resolved_anchor = vertical_anchor
            break
    if resolved_anchor is None:
        valid_values = [vertical_anchor.value for vertical_anchor in VerticalAnchor]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"Invalid vertical anchor: {anchor!r}."
            f" Expected one of {valid_string}.",
        )
    return resolved_anchor


def as_horizontal_anchor(
    anchor: HorizontalAnchorLike,
) -> HorizontalAnchor:
    """
    Convert into a canonical HorizontalAnchor.

    Accepts either:
    - Enum members: HorizontalAnchor.Left / Center / Right
    - Strings matching the Enum value (case insensitive)
    - Strings matching the Enum name (case insensitive)

    Returns one of:
    - HorizontalAnchor.Left
    - HorizontalAnchor.Center
    - HorizontalAnchor.Right
    """
    if isinstance(anchor, HorizontalAnchor):
        return anchor
    if not isinstance(anchor, str):
        raise TypeError(
            f"Expected a HorizontalAnchor or str; got {type(anchor).__name__}.",
        )
    anchor_lower = anchor.lower()
    resolved_anchor = None
    for horizontal_anchor in HorizontalAnchor:
        is_like_anchor_value = (anchor_lower == horizontal_anchor.value.lower())
        is_like_anchor_name = (anchor_lower == horizontal_anchor.name.lower())
        if is_like_anchor_value or is_like_anchor_name:
            resolved_anchor = horizontal_anchor
            break
    if resolved_anchor is None:
        valid_values = [horizontal_anchor.value for horizontal_anchor in HorizontalAnchor]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"Invalid horizontal anchor: {anchor!r}."
            f" Expected one of {valid_string}.",
        )
    return resolved_anchor


def as_vertical_edge_anchor(
    anchor: VerticalAnchorLike,
    *,
    param_name: str = "<param>",
) -> VerticalAnchor:
    """Convert into a VerticalAnchor and ensure it is an edge (Top/Bottom, not Center)."""
    resolved_anchor = as_vertical_anchor(anchor)
    ensure_vertical_edge_anchor(
        anchor=resolved_anchor,
        param_name=param_name,
    )
    return resolved_anchor


def as_horizontal_edge_anchor(
    anchor: HorizontalAnchorLike,
    *,
    param_name: str = "<param>",
) -> HorizontalAnchor:
    """Convert into a HorizontalAnchor and ensure it is an edge (Left/Right, not Center)."""
    resolved_anchor = as_horizontal_anchor(anchor)
    ensure_horizontal_edge_anchor(
        anchor=resolved_anchor,
        param_name=param_name,
    )
    return resolved_anchor


def as_edge_anchor(
    anchor: AnchorLike,
    *,
    param_name: str = "<param>",
) -> VerticalAnchor | HorizontalAnchor:
    """
    Convert into an edge VerticalAnchor or HorizontalAnchor.

    Explicit logic, no try/except guessing:

    - If `anchor` is already a VerticalAnchor / HorizontalAnchor:
        - enforce edge and return it.
    - If `anchor` is a string:
        - for 'top'/'center'/'bottom' → treat as vertical, enforce edge.
        - for 'left'/'center'/'right' → treat as horizontal, enforce edge.
        - anything else → ValueError.
    """
    if isinstance(anchor, VerticalAnchor):
        ensure_vertical_edge_anchor(
            anchor=anchor,
            param_name=param_name,
        )
        return anchor
    if isinstance(anchor, HorizontalAnchor):
        ensure_horizontal_edge_anchor(
            anchor=anchor,
            param_name=param_name,
        )
        return anchor
    if isinstance(anchor, str):
        anchor_lower = anchor.lower()
        if anchor_lower in {vertical_anchor.value for vertical_anchor in VerticalAnchor}:
            resolved_anchor = as_vertical_anchor(anchor_lower)
            ensure_vertical_edge_anchor(
                anchor=resolved_anchor,
                param_name=param_name,
            )
            return resolved_anchor
        if anchor_lower in {horizontal_anchor.value for horizontal_anchor in HorizontalAnchor}:
            resolved_anchor = as_horizontal_anchor(anchor_lower)
            ensure_horizontal_edge_anchor(
                anchor=resolved_anchor,
                param_name=param_name,
            )
            return resolved_anchor
        valid_values = [
            VerticalAnchor.Top.value,
            VerticalAnchor.Bottom.value,
            HorizontalAnchor.Left.value,
            HorizontalAnchor.Right.value,
        ]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"Invalid edge anchor string: {anchor!r} in `{param_name}`."
            f" Expected one of {valid_string}.",
        )
    raise TypeError(
        f"`{param_name}` must be a VerticalAnchor, HorizontalAnchor, or str;"
        f" got {type(anchor).__name__}.",
    )


##
## === TYPE VALIDATORS
##


def ensure_vertical_anchor(
    anchor: VerticalAnchor,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `anchor` is a VerticalAnchor."""
    if not isinstance(anchor, VerticalAnchor):
        raise TypeError(
            f"`{param_name}` must be a VerticalAnchor; got {type(anchor).__name__}.",
        )


def ensure_horizontal_anchor(
    anchor: HorizontalAnchor,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `anchor` is a HorizontalAnchor."""
    if not isinstance(anchor, HorizontalAnchor):
        raise TypeError(
            f"`{param_name}` must be a HorizontalAnchor; got {type(anchor).__name__}.",
        )


def ensure_vertical_edge_anchor(
    anchor: VerticalAnchor,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `anchor` is a vertical edge anchor (Top/Bottom, not Center)."""
    ensure_vertical_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    if anchor is VerticalAnchor.Center:
        valid_values = [
            VerticalAnchor.Top.value,
            VerticalAnchor.Bottom.value,
        ]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"`{param_name}` must be a vertical edge anchor but got {anchor!r}."
            f" Expected one of {valid_string}.",
        )


def ensure_horizontal_edge_anchor(
    anchor: HorizontalAnchor,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `anchor` is a horizontal edge anchor (Left/Right, not Center)."""
    ensure_horizontal_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    if anchor is HorizontalAnchor.Center:
        valid_values = [
            HorizontalAnchor.Left.value,
            HorizontalAnchor.Right.value,
        ]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"`{param_name}` must be a horizontal edge anchor but got {anchor!r}."
            f" Expected one of {valid_string}.",
        )


##
## === TYPE CHECKS
##


def is_top_edge(
    anchor: VerticalAnchor,
    *,
    param_name: str = "<param>",
) -> bool:
    ensure_vertical_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    return anchor is VerticalAnchor.Top


def is_bottom_edge(
    anchor: VerticalAnchor,
    *,
    param_name: str = "<param>",
) -> bool:
    ensure_vertical_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    return anchor is VerticalAnchor.Bottom


def is_left_edge(
    anchor: HorizontalAnchor,
    *,
    param_name: str = "<param>",
) -> bool:
    ensure_horizontal_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    return anchor is HorizontalAnchor.Left


def is_right_edge(
    anchor: HorizontalAnchor,
    *,
    param_name: str = "<param>",
) -> bool:
    ensure_horizontal_anchor(
        anchor=anchor,
        param_name=param_name,
    )
    return anchor is HorizontalAnchor.Right


## } MODULE
