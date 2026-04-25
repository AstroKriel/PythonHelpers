## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import types
from enum import Enum
from typing import get_args

## local
from jormi import ww_lists
from jormi.ww_checks import check_python_types
from jormi.ww_types import python_enums

##
## === INTERNAL HELPERS
##


def _normalise_string(
    string: str,
) -> str:
    return string.strip().lower()


def _find_match_in_enum(
    member_key: str,
    *,
    enum_type: python_enums.EnumType,
) -> Enum | None:
    for member in enum_type:
        if member_key == _normalise_string(member.name):
            return member
        if member_key == _normalise_string(str(member.value)):
            return member
    return None


def _find_unique_match(
    member_key: str,
    *,
    valid_enums: tuple[python_enums.EnumType, ...],
) -> Enum | None:
    matched_member = None
    for enum_type in valid_enums:
        candidate_member = _find_match_in_enum(
            member_key=member_key,
            enum_type=enum_type,
        )
        if candidate_member is None:
            continue
        if (matched_member is not None) and (candidate_member is not matched_member):
            raise ValueError("ambiguous Enum member.")
        matched_member = candidate_member
    return matched_member


def _enum_member_names(
    enum_types: tuple[python_enums.EnumType, ...],
) -> str:
    member_names = [member.name for enum_type in enum_types for member in enum_type]
    member_names = sorted(set(member_names))
    return ww_lists.as_quoted_string(member_names)


##
## === INTERACTING WITH ENUMS + THEIR MEMBERS
##


def as_runtime_type(
    type_hint: type[Enum] | types.UnionType,
) -> tuple[type[Enum], ...]:
    """
    Convert a union of Enum member types into a tuple of Enum classes
    suitable for `check_enums`' runtime validation.
    """
    args = get_args(type_hint)
    if args:
        if not all(isinstance(arg, type) and issubclass(arg, Enum) for arg in args):
            raise TypeError(f"non-Enum argument(s) in hint: {type_hint!r}.")
        return tuple(args)
    if isinstance(type_hint, type) and issubclass(
            type_hint,
            Enum,
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        return (type_hint, )
    raise TypeError(f"unsupported Enum type-hint: {type_hint!r}.")


def ensure_sequence_of_enums(
    param: tuple[python_enums.EnumType, ...] | list[python_enums.EnumType],
    *,
    param_name: str = "param",
) -> None:
    """Ensure `param` is a non-empty sequence of Enum types."""
    check_python_types.ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        valid_seq_types=check_python_types.RuntimeTypes.Sequences.SequenceLike,
        valid_elem_types=type,
    )
    ## reject empty sequences
    if not param:
        raise ValueError(f"`{param_name}` must be non-empty.")
    ## reject sequences containing non-Enum types
    if not all(issubclass(enum_type, Enum)  # pyright: ignore[reportUnnecessaryIsInstance]
               for enum_type in param):
        raise TypeError(f"all `{param_name}` entries must be Enum types.")


def resolve_member(
    member: python_enums.EnumMemberLike,
    *,
    valid_enums: python_enums.EnumTypesLike,
) -> Enum:
    """Return `member` as an Enum member from one of `valid_enums`."""
    valid_enums = check_python_types.as_tuple(
        param=valid_enums,
        param_name="valid_enums",
    )
    ensure_sequence_of_enums(
        param=valid_enums,
        param_name="valid_enums",
    )
    ## Enum members are themselves an Enum (an instance of the same parent Enum)
    ## if member is a valid Enum, return
    if isinstance(member, Enum):
        if isinstance(member, valid_enums):
            return member
        raise ValueError(f"enum member {member!r} is not in the set of valid Enum types.")
    ## otherwise search for a unique instance of the string the user passed in valid_enums name or value
    check_python_types.ensure_type(
        param=member,
        param_name="member",
        valid_types=str,
    )
    normalised_member = _normalise_string(member)
    matched_member = _find_unique_match(
        member_key=normalised_member,
        valid_enums=valid_enums,
    )
    if matched_member is not None:
        return matched_member
    raise ValueError(
        f"Invalid Enum member: {member!r}; expected one of: {_enum_member_names(valid_enums)} (to match by member name).",
    )


def ensure_valid_member(
    member: python_enums.EnumMemberLike,
    *,
    valid_enums: python_enums.EnumTypesLike,
    param_name: str = "<param>",
) -> None:
    try:
        resolve_member(
            member=member,
            valid_enums=valid_enums,
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"`{param_name}` is invalid.") from error


def ensure_member_in(
    member: python_enums.EnumMemberLike,
    *,
    valid_members: tuple[Enum, ...] | list[Enum],
    param_name: str = "<param>",
) -> None:
    valid_members = check_python_types.as_tuple(
        param=valid_members,
        param_name="valid_members",
    )
    if not valid_members:
        raise ValueError("`valid_members` must be non-empty.")
    if not all(isinstance(member, Enum)  # pyright: ignore[reportUnnecessaryIsInstance]
               for member in valid_members):
        raise TypeError("`valid_members` entries must be Enum members.")
    valid_enums = tuple({type(member) for member in valid_members})
    resolved_member = resolve_member(
        member=member,
        valid_enums=valid_enums,
    )
    if resolved_member not in valid_members:
        valid_members_string = ww_lists.as_quoted_string([member.name for member in valid_members])
        raise ValueError(
            f"`{param_name}` must be one of: {valid_members_string}; got {resolved_member.name}.",
        )


## } MODULE
