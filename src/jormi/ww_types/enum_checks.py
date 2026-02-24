## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from jormi.ww_types import type_checks

##
## === TYPE DEFINITIONS
## Enumerators (Enums for short) are a class (i.e. type) that defines a fixed set of named members;
## each member has a `.name` and `.value` are themselves also an instance of the Enum
##

EnumType = type[Enum]
EnumTypesLike = EnumType | tuple[EnumType, ...] | list[EnumType]
EnumMemberLike = Enum | str

##
## === HELPER FUNCTIONS
##


def _ensure_enums(
    param: EnumTypesLike,
    *,
    param_name: str = "param",
) -> None:
    """Ensure `param` is an Enum class, or a non-empty sequence of Enum types."""
    ## allow a single Enum
    if isinstance(param, type):
        if not issubclass(param, Enum):
            raise TypeError(f"`{param_name}` must be an Enum class.")
        return
    ## otherwise require a flat sequence (e.g., tuple or list) of types (Enums are a subset of types)
    type_checks.ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        valid_seq_types=type_checks.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=type,
    )
    ## reject empty sequences
    if not param:
        raise ValueError(f"`{param_name}` must be non-empty.")
    ## reject sequences containing non-Enum types
    if not all(issubclass(enum_type, Enum) for enum_type in param):
        raise TypeError(f"`{param_name}` all entries must be Enum types.")


def _normalise_string(
    string: str,
) -> str:
    return string.strip().lower()


def _find_match_in_enum(
    key: str,
    *,
    enum_type: EnumType,
) -> Enum | None:
    for member in enum_type:
        if key == _normalise_string(member.name):
            return member
        if key == _normalise_string(str(member.value)):
            return member
    return None


def _find_unique_match(
    key: str,
    *,
    valid_enums: tuple[EnumType, ...],
) -> Enum | None:
    match = None
    for enum_type in valid_enums:
        candidate = _find_match_in_enum(
            key=key,
            enum_type=enum_type,
        )
        if candidate is None:
            continue
        if (match is not None) and (candidate is not match):
            raise ValueError("Ambiguous Enum member.")
        match = candidate
    return match


def _resolve_all_enum_values(
    valid_enums: tuple[EnumType, ...],
) -> str:
    values = [str(member.value) for enum_type in valid_enums for member in enum_type]
    values = sorted(set(values))
    return ", ".join(repr(value) for value in values)


def resolve_member(
    member: EnumMemberLike,
    *,
    valid_enums: EnumTypesLike,
) -> Enum:
    """Return `member` as an Enum member from one of `valid_enums`."""
    valid_enums = type_checks.as_tuple(
        param=valid_enums,
        param_name="valid_enums",
    )
    _ensure_enums(
        param=valid_enums,
        param_name="valid_enums",
    )
    ## Enum members are themselves an Enum (an instance of the same parent Enum)
    ## if member is a valid Enum, return
    if isinstance(member, Enum):
        if isinstance(member, valid_enums):
            return member
        raise ValueError(f"Enum member {member!r} is not in the set of valid Enum types.")
    ## otherwise search for a unique instance of the string the user passed in valid_enums name or value
    type_checks.ensure_type(
        param=member,
        param_name="member",
        valid_types=str,
    )
    key = _normalise_string(member)
    match = _find_unique_match(
        key=key,
        valid_enums=valid_enums,
    )
    if match is not None:
        return match
    raise ValueError(
        f"Invalid Enum member: {member!r}; expected one of: {_resolve_all_enum_values(valid_enums)}.",
    )


def ensure_valid_member(
    member: EnumMemberLike,
    *,
    valid_enums: EnumTypesLike,
    param_name: str = "<param>",
) -> None:
    try:
        resolve_member(
            member=member,
            valid_enums=valid_enums,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"`{param_name}` must be a valid Enum member; got {member!r} ({type(member).__name__}).",
        ) from error


def ensure_member_in(
    member: EnumMemberLike,
    *,
    valid_members: tuple[Enum, ...] | list[Enum],
    param_name: str = "<param>",
) -> None:
    valid_members = type_checks.as_tuple(
        param=valid_members,
        param_name="valid_members",
    )
    if not valid_members:
        raise ValueError("`valid_members` must be non-empty.")
    if not all(isinstance(member, Enum) for member in valid_members):
        raise TypeError("`valid_members` entries must be Enum members.")
    valid_enums = tuple({type(member) for member in valid_members})
    resolved_member = resolve_member(
        member=member,
        valid_enums=valid_enums,
    )
    if resolved_member not in valid_members:
        valid_members_string = ", ".join(member.name for member in valid_members)
        raise ValueError(f"`{param_name}` must be one of: {valid_members_string}; got {resolved_member.name}.")


## } MODULE
