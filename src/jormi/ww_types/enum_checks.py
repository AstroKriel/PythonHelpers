## { MODULE

##
## === DEPENDENCIES
##

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

from jormi.utils import list_utils
from jormi.ww_types import type_checks

##
## === TYPE DEFINITIONS
##


EnumType: TypeAlias = type[Enum]
EnumTypesLike: TypeAlias = EnumType | tuple[EnumType, ...] | list[EnumType]
EnumMemberLike: TypeAlias = Enum | str

##
## === TYPE NORMALISERS
##


def as_tuple_of_enum_types(
    valid_types: EnumTypesLike,
    *,
    param_name: str = "valid_types",
) -> tuple[EnumType, ...]:
    """
    Canonicalize `valid_types` into a non-empty tuple of Enum subclasses.

    Accepts:
      - A single Enum class
      - A tuple/list of Enum classes

    Returns:
      - tuple of Enum classes
    """
    valid_types_tuple = type_checks.as_tuple(
        param=valid_types,
        param_name=param_name,
    )

    if len(valid_types_tuple) == 0:
        raise ValueError(f"`{param_name}` must be non-empty.")

    bad_types = [
        valid_type for valid_type in valid_types_tuple
        if (not isinstance(valid_type, type)) or (not issubclass(valid_type, Enum))
    ]
    if bad_types:
        bad_type_names = ", ".join(
            getattr(bad_type, "__name__", repr(bad_type))
            for bad_type in bad_types
        )
        raise TypeError(
            f"`{param_name}` must contain Enum types only; got {bad_type_names}.",
        )

    return tuple(valid_types_tuple)


##
## === ENUM RESOLUTION
##


def as_enum_member(
    member: EnumMemberLike,
    *,
    valid_types: EnumTypesLike,
) -> Enum:
    """
    Resolve `member` into a canonical Enum member drawn from `valid_types`.

    Accepts:
      - Enum members from one of `valid_types`
      - Strings matching an Enum member's value (case-insensitive)
      - Strings matching an Enum member's name  (case-insensitive)

    Returns:
      - A member of one of the enums in `valid_types`.
    """
    valid_types_tuple = as_tuple_of_enum_types(
        valid_types=valid_types,
        param_name="valid_types",
    )

    if isinstance(member, Enum):
        for enum_type in valid_types_tuple:
            if isinstance(member, enum_type):
                return member

    member_lower = str(member).strip().lower()

    for enum_type in valid_types_tuple:
        for candidate_member in enum_type:
            is_like_value = (member_lower == str(candidate_member.value).lower())
            is_like_name = (member_lower == candidate_member.name.lower())
            if is_like_value or is_like_name:
                return candidate_member

    valid_values = [
        str(candidate_member.value)
        for enum_type in valid_types_tuple
        for candidate_member in enum_type
    ]
    valid_values_string = list_utils.as_quoted_string(elems=sorted(valid_values))
    raise ValueError(
        f"Invalid enum member: {member!r}."
        f" Expected one of {valid_values_string}.",
    )


def ensure_enum_member(
    member: EnumMemberLike,
    *,
    valid_types: EnumTypesLike,
    param_name: str = "<param>",
) -> None:
    """Ensure `member` can be resolved into one of `valid_types`."""
    try:
        as_enum_member(
            member=member,
            valid_types=valid_types,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"`{param_name}` must be resolvable to one of the allowed enum types; "
            f"got {member!r} ({type(member).__name__}).",
        ) from error


##
## === SUBSET VALIDATION
##


def ensure_enum_member_in(
    member: EnumMemberLike,
    *,
    valid_members: tuple[Enum, ...] | list[Enum],
    param_name: str = "<param>",
) -> None:
    """
    Ensure `member` resolves to one of `valid_members`.

    This is useful for enforcing axis-specific subsets (e.g. Top/Bottom only).
    """
    valid_members_tuple = type_checks.as_tuple(
        param=valid_members,
        param_name="valid_members",
    )
    if len(valid_members_tuple) == 0:
        raise ValueError("`valid_members` must be non-empty.")

    valid_types = tuple({type(valid_member) for valid_member in valid_members_tuple})
    resolved_member = as_enum_member(
        member=member,
        valid_types=valid_types,
    )

    if resolved_member not in valid_members_tuple:
        valid_names = [valid_member.name for valid_member in valid_members_tuple]
        valid_string = list_utils.as_quoted_string(elems=sorted(valid_names))
        raise ValueError(
            f"`{param_name}` must be one of {valid_string}; got {resolved_member.name}.",
        )


## } MODULE