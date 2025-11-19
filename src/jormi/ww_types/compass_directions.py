## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from typing import TypeAlias

from jormi.utils import list_utils

##
## === DATA TYPES
##


class CompassCardinal(str, Enum):
    """Cardinal compass directions in 2D index space."""

    NORTH = "n"
    SOUTH = "s"
    EAST = "e"
    WEST = "w"


class CompassOrdinal(str, Enum):
    """Ordinal (inter-cardinal) compass directions in 2D index space."""

    NORTHEAST = "ne"
    SOUTHEAST = "se"
    SOUTHWEST = "sw"
    NORTHWEST = "nw"


CompassDirection: TypeAlias = CompassCardinal | CompassOrdinal

CompassCardinalLike: TypeAlias = CompassCardinal | str
CompassOrdinalLike: TypeAlias = CompassOrdinal | str
CompassDirectionLike: TypeAlias = CompassDirection | str

##
## === TYPE VALIDATORS
##


def as_compass_cardinal(
    direction: CompassCardinalLike,
) -> CompassCardinal:
    """
    Convert a direction into a CompassCardinal.

    Accepts either:
    - Enum members: CompassCardinal.NORTH
    - Strings matching the Enum value (e.g. "n", "N")
    - Strings matching the Enum name (e.g. "north", "NORTH")
    """
    if isinstance(direction, CompassCardinal):
        return direction
    direction_lower = direction.lower()
    for compass_cardinal in CompassCardinal:
        is_like_value = (direction_lower == compass_cardinal.value.lower())
        is_like_name = (direction_lower == compass_cardinal.name.lower())
        if is_like_value or is_like_name:
            return compass_cardinal
    valid_values = [compass_direction.value for compass_direction in CompassCardinal]
    valid_string = list_utils.as_string(
        elems=valid_values,
        wrap_in_quotes=True,
        conjunction="",
    )
    raise ValueError(
        f"Invalid compass cardinal direction: {direction!r}."
        f" Expected one of {valid_string}.",
    )


def as_compass_ordinal(
    direction: CompassOrdinalLike,
) -> CompassOrdinal:
    """
    Convert a direction into a CompassOrdinal.

    Accepts either:
    - Enum members: CompassOrdinal.NORTHEAST
    - Strings matching the Enum value (e.g. "ne", "NE")
    - Strings matching the Enum name (e.g. "northeast", "NORTHEAST")
    """
    if isinstance(direction, CompassOrdinal):
        return direction
    direction_lower = direction.lower()
    for compass_ordinal in CompassOrdinal:
        is_like_value = (direction_lower == compass_ordinal.value.lower())
        is_like_name = (direction_lower == compass_ordinal.name.lower())
        if is_like_value or is_like_name:
            return compass_ordinal
    valid_values = [compass_direction.value for compass_direction in CompassOrdinal]
    valid_string = list_utils.as_string(
        elems=valid_values,
        wrap_in_quotes=True,
        conjunction="",
    )
    raise ValueError(
        f"Invalid compass ordinal direction: {direction!r}."
        f" Expected one of {valid_string}.",
    )


def as_compass_direction(
    direction: CompassDirectionLike,
) -> CompassDirection:
    """
    Convert a direction into a CompassDirection.

    Accepts either:
    - Enum members: CompassCardinal / CompassOrdinal
    - Strings matching any Enum value (e.g. "n", "ne")
    - Strings matching any Enum name (e.g. "north", "northeast")
    """
    if isinstance(direction, (CompassCardinal, CompassOrdinal)):
        return direction
    try:
        return as_compass_cardinal(direction)
    except ValueError:
        pass
    try:
        return as_compass_ordinal(direction)
    except ValueError:
        pass
    valid_values = [compass_direction.value for compass_direction in (*CompassCardinal, *CompassOrdinal)]
    valid_string = list_utils.as_string(
        elems=valid_values,
        wrap_in_quotes=True,
        conjunction="",
    )
    raise ValueError(
        f"Invalid compass direction: {direction!r}."
        f" Expected one of {valid_string}.",
    )


## } MODULE
