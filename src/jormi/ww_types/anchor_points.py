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

##
## === CONVERTERS
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


## } MODULE
