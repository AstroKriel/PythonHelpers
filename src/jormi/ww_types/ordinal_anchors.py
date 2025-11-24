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


class CornerAnchor(str, Enum):
    """
    Discrete corner/edge anchors for legend-like boxes.

    Values are chosen to match Matplotlib's `loc` strings.
    """

    TopLeft = "upper left"
    TopRight = "upper right"
    BottomLeft = "lower left"
    BottomRight = "lower right"
    Center = "center"
    TopCenter = "upper center"
    BottomCenter = "lower center"
    CenterLeft = "center left"
    CenterRight = "center right"


CornerAnchorLike: TypeAlias = CornerAnchor | str

##
## === TYPE CONVERTER
##


def as_corner_anchor(
    anchor: CornerAnchorLike,
) -> CornerAnchor:
    """
    Convert into a canonical CornerAnchor.

    Accepts either:
    - Enum members: CornerAnchor.TopLeft / TopRight / ...
    - Strings matching the Enum value (case insensitive)
    - Strings matching the Enum name (case insensitive)

    Returns one of the CornerAnchor members.
    """
    if isinstance(anchor, CornerAnchor):
        return anchor
    anchor_lower = str(anchor).lower()
    resolved_anchor = None
    for corner_anchor in CornerAnchor:
        is_like_anchor_value = (anchor_lower == corner_anchor.value.lower())
        is_like_anchor_name = (anchor_lower == corner_anchor.name.lower())
        if is_like_anchor_value or is_like_anchor_name:
            resolved_anchor = corner_anchor
            break
    if resolved_anchor is None:
        valid_values = [corner_anchor.value for corner_anchor in CornerAnchor]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"Invalid corner anchor: {anchor!r}."
            f" Expected one of {valid_string}.",
        )
    return resolved_anchor


##
## === TYPE VALIDATORS
##


def ensure_corner_anchor(
    anchor: CornerAnchor,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `anchor` is a CornerAnchor."""
    if not isinstance(anchor, CornerAnchor):
        raise TypeError(
            f"`{param_name}` must be a CornerAnchor; got {type(anchor).__name__}.",
        )


## } MODULE
