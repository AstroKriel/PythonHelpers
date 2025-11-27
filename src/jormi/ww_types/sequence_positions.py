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


class SequencePosition(str, Enum):
    """Logical position within a 1D sequence."""

    First = "first"
    Middle = "middle"
    Last = "last"


SequencePositionLike: TypeAlias = SequencePosition | str


##
## === TYPE CONVERTER
##


def as_sequence_position(
    position: SequencePositionLike,
) -> SequencePosition:
    """
    Convert into a canonical SequencePosition.

    Accepts either:
    - Enum members: SequencePosition.First / Middle / Last
    - Strings matching the Enum value (case insensitive)
    - Strings matching the Enum name (case insensitive)

    Returns one of the SequencePosition members.
    """
    if isinstance(position, SequencePosition):
        return position
    position_lower = str(position).lower()
    resolved_position = None
    for seq_position in SequencePosition:
        is_like_value = (position_lower == seq_position.value.lower())
        is_like_name = (position_lower == seq_position.name.lower())
        if is_like_value or is_like_name:
            resolved_position = seq_position
            break
    if resolved_position is None:
        valid_values = [seq_position.value for seq_position in SequencePosition]
        valid_string = list_utils.as_quoted_string(elems=valid_values)
        raise ValueError(
            f"Invalid sequence position: {position!r}."
            f" Expected one of {valid_string}.",
        )
    return resolved_position


##
## === TYPE VALIDATORS
##


def ensure_sequence_position(
    position: SequencePosition,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `position` is a SequencePosition."""
    if not isinstance(position, SequencePosition):
        raise TypeError(
            f"`{param_name}` must be a SequencePosition; got {type(position).__name__}.",
        )


## } MODULE
