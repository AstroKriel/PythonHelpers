## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from enum import Enum
from typing import TypeAlias

## local
from jormi.ww_types import enums

##
## === POSITIONS
##


class Positions:
    """
    Public namespace for box position enums; use for both type annotations and value selection.

    For mpl-specific usage, prefer `MPLPositions`, which pre-filters members to what each mpl
    parameter actually accepts.
    """

    PositionLike: TypeAlias = enums.EnumMemberLike

    class Center(str, Enum):
        Center = "center"

    class Corner(str, Enum):
        TopLeft = "upper left"
        TopRight = "upper right"
        BottomLeft = "lower left"
        BottomRight = "lower right"

    class Edge(str, Enum):
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
## === MPL POSITIONS
##


class MPLPositions:
    """
    `Positions` organised by mpl parameter concept.

    Use instead of `Positions` when selecting a value for a specific mpl parameter; the
    sub-class pre-filters members to what that parameter actually accepts, so no knowledge
    of the underlying geometry types is required.
    """

    class Anchor:
        """Valid positions for mpl `loc`."""

        Corner = Positions.Corner
        Edge = Positions.Edge
        Center = Positions.Center

    class Align:
        """Valid positions for mpl `ha` and `va` (text horizontal and vertical alignment)."""

        Side = Positions.Side
        Center = Positions.Center


##
## === TYPE ALIASES
## mpl-specific subsets; private because callers select from Positions.* members.
##

_AnchorLike: TypeAlias = Positions.Corner | Positions.Edge | Positions.Center
_AlignLike: TypeAlias = Positions.Side | Positions.Center

## } MODULE
