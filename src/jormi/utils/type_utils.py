## { MODULE

##
## === FUNCTIONS
##


def _as_tuple(
    valid_types: type | tuple[type, ...] | list[type],
) -> tuple[type, ...]:
    """Convert a type or sequence of types into a canonical (type, ...) tuple."""
    if isinstance(valid_types, type):
        return (valid_types, )
    if isinstance(valid_types, (tuple, list)):
        if not valid_types:
            raise ValueError("Empty type specification.")
        if not all(isinstance(elem_type, type) for elem_type in valid_types):
            raise TypeError("`valid_types` entries must be valid Python types.")
        return tuple(valid_types)
    raise TypeError("`valid_types` must be a type or a tuple/list of types.")


def assert_type(
    *,
    var_obj,
    valid_types: type | tuple[type, ...] | list[type],
    var_name: str = "<var>",
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is an instance of the required type(s)."""
    if var_obj is None and allow_none: return
    valid_types = _as_tuple(valid_types)
    if not isinstance(var_obj, valid_types):
        type_names = ", ".join(elem_type.__name__ for elem_type in valid_types)
        raise TypeError(f"`{var_name}` is {type(var_obj).__name__}; expected {type_names}.")


def assert_not_none(
    *,
    var_obj,
    var_name: str = "<var>",
) -> None:
    """Ensure a variable is not None."""
    if var_obj is None:
        raise ValueError(f"`{var_name}` must not be None.")


def assert_sequence(
    *,
    var_obj,
    var_name: str = "<seq>",
    valid_containers: tuple[type, ...] = (tuple, list),
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is a valid sequence container, with optional fixed length and uniform element types."""
    if var_obj is None and allow_none: return
    ## enforce container type
    valid_containers = _as_tuple(valid_containers)
    if not isinstance(var_obj, valid_containers):
        allowed_str = ", ".join(valid_type.__name__ for valid_type in valid_containers)
        raise TypeError(f"`{var_name}` must be one of ({allowed_str}).")
    ## enforce number of elements
    if (seq_length is not None) and (len(var_obj) != seq_length):
        raise ValueError(f"`{var_name}` must have length {seq_length} (got {len(var_obj)}).")
    ## enforce uniform element types
    if valid_elem_types is not None:
        valid_elem_types = _as_tuple(valid_elem_types)
        bad_indices = [
            elem_index for elem_index, elem in enumerate(var_obj) if not isinstance(elem, valid_elem_types)
        ]
        if bad_indices:
            preview_str = ", ".join(map(str, bad_indices[:5])) + ("..." if len(bad_indices) > 5 else "")
            required_str = ", ".join(valid_type.__name__ for valid_type in valid_elem_types)
            raise TypeError(
                f"`{var_name}` elements must be of type(s) {required_str}; failed at indices: {preview_str}.",
            )


def assert_nonempty_str(
    *,
    var_obj: str,
    var_name: str = "<var>",
) -> None:
    """Enforce a non-empty string."""
    assert_type(
        var_obj=var_obj,
        valid_types=str,
        var_name=var_name,
    )
    if not var_obj:
        raise ValueError(f"`{var_name}` must be a non-empty string.")


## } MODULE
