## { MODULE
##
## === DEPENDENCIES
##

import numpy

##
## === FUNCTIONS
##


def as_tuple(
    *,
    seq_obj,
    var_name: str = "<seq>",
) -> tuple:
    """Return `seq_obj` as a tuple."""
    if seq_obj is None:
        raise ValueError(f"`{var_name}` must not be None.")
    if isinstance(seq_obj, tuple):
        return seq_obj
    if isinstance(seq_obj, list):
        return tuple(seq_obj)
    return (seq_obj, )


def _types_to_tuple(
    valid_types: type | tuple[type, ...] | list[type],
) -> tuple[type, ...]:
    """Canonicalize a type or sequence of types to a (type, ...) tuple."""
    if isinstance(valid_types, type):
        return (valid_types, )
    if isinstance(valid_types, (tuple, list)):
        if not valid_types:
            raise ValueError("Empty type specification.")
        if not all(isinstance(valid_type, type) for valid_type in valid_types):
            raise TypeError("`valid_types` entries must be valid Python types.")
        return tuple(valid_types)
    raise TypeError("`valid_types` must be a type or a tuple/list of types.")


def ensure_type(
    *,
    var_obj,
    valid_types: type | tuple[type, ...] | list[type],
    var_name: str = "<var>",
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is an instance of the required type(s)."""
    if (var_obj is None) and allow_none:
        return
    valid_types = _types_to_tuple(valid_types)
    if not isinstance(var_obj, valid_types):
        valid_types_str = ", ".join(valid_type.__name__ for valid_type in valid_types)
        raise TypeError(f"`{var_name}` is {type(var_obj).__name__}; expected {valid_types_str}.")


def ensure_not_none(
    *,
    var_obj,
    var_name: str = "<var>",
) -> None:
    """Ensure a variable is not None."""
    if var_obj is None:
        raise ValueError(f"`{var_name}` must not be None.")


def ensure_sequence(
    *,
    var_obj,
    var_name: str = "<seq>",
    valid_containers: tuple[type, ...] = (tuple, list),
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is a valid sequence container, with optional fixed length and uniform element types."""
    if (var_obj is None) and allow_none:
        return
    ## enforce container type
    valid_containers = _types_to_tuple(valid_containers)
    if not isinstance(var_obj, valid_containers):
        valid_container_str = ", ".join(valid_container.__name__ for valid_container in valid_containers)
        raise TypeError(f"`{var_name}` must be one of ({valid_container_str}).")
    ## enforce number of elements
    if (seq_length is not None) and (len(var_obj) != seq_length):
        raise ValueError(f"`{var_name}` must have length {seq_length} (got {len(var_obj)}).")
    ## enforce uniform element types
    if valid_elem_types is not None:
        valid_elem_types = _types_to_tuple(valid_elem_types)
        bad_indices = [
            elem_index for elem_index, elem in enumerate(var_obj) if not isinstance(elem, valid_elem_types)
        ]
        if bad_indices:
            preview_bad_indices_str = ", ".join(
                map(str, bad_indices[:5]),
            ) + ("..." if len(bad_indices) > 5 else "")
            valid_elem_types_str = ", ".join(valid_elem_type.__name__ for valid_elem_type in valid_elem_types)
            raise TypeError(
                f"`{var_name}` elements must be of type(s) {valid_elem_types_str}; failed at indices: {preview_bad_indices_str}.",
            )


def ensure_nonempty_str(
    *,
    var_obj: str,
    var_name: str = "<var>",
) -> None:
    """Enforce a non-empty string."""
    ensure_type(
        var_obj=var_obj,
        valid_types=str,
        var_name=var_name,
    )
    if not var_obj:
        raise ValueError(f"`{var_name}` must be a non-empty string.")


def ensure_finite_number(
    *,
    var_obj,
    var_name: str,
    valid_types: tuple[type, ...],
    allow_none: bool,
) -> None:
    """Internal helper: ensure `var_obj` is finite and of the given numeric types."""
    if var_obj is None:
        if allow_none: return
        raise ValueError(f"`{var_name}` must not be None.")
    ## reject booleans explicitly (they are subclasses of int)
    if isinstance(var_obj, (bool, numpy.bool_)):
        raise TypeError(f"`{var_name}` must not be a boolean.")
    if not isinstance(var_obj, valid_types):
        expected = ", ".join(t.__name__ for t in valid_types)
        raise TypeError(f"`{var_name}` is {type(var_obj).__name__}; expected {expected}.")
    if not numpy.isfinite(var_obj):
        raise ValueError(f"`{var_name}` must be finite (got {var_obj}).")


def ensure_finite_float(
    *,
    var_obj,
    var_name: str = "<var>",
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is a finite float-like value."""
    ensure_finite_number(
        var_obj=var_obj,
        var_name=var_name,
        valid_types=(float, numpy.floating),
        allow_none=allow_none,
    )


def ensure_finite_int(
    *,
    var_obj,
    var_name: str = "<var>",
    allow_none: bool = False,
) -> None:
    """Ensure `var_obj` is a finite int-like value."""
    ensure_finite_number(
        var_obj=var_obj,
        var_name=var_name,
        valid_types=(int, numpy.integer),
        allow_none=allow_none,
    )


## } MODULE
