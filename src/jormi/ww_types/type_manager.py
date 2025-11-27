## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import get_args

##
## === TYPE DEFINITIONS
##
## TypeHints defines conceptual categories for annotations.
## RuntimeTypes converts those into tuples for isinstance checks.
##


class TypeHints:
    """Type-hint groupings for common concepts."""

    class Strings:
        StringLike = str

    class Booleans:
        BooleanLike = bool | numpy.bool_

    class Numerics:
        IntLike = int | numpy.integer
        FloatLike = float | numpy.floating
        NumericLike = IntLike | FloatLike

    class Containers:
        SetLike = set
        DictLike = dict
        ArrayLike = numpy.ndarray
        ContainerLike = SetLike | DictLike | ArrayLike

    class Sequences:
        ListLike = list
        TupleLike = tuple
        SequenceLike = ListLike | TupleLike


def _as_runtime_type(
    type_hint,
) -> tuple[type, ...]:
    """
    Convert a type-hint alias (possibly a union) into a tuple of runtime
    types suitable for isinstance checks.
    """
    args = get_args(type_hint)
    if args:
        if not all(isinstance(arg, type) for arg in args):
            raise TypeError(f"Non-type argument(s) in hint: {type_hint!r}")
        return tuple(args)
    if isinstance(type_hint, type):
        return (type_hint, )
    raise TypeError(f"Unsupported type-hint: {type_hint!r}")


class RuntimeTypes:
    """Runtime type tuples derived from TypeHints for isinstance checks."""

    class Strings:
        StringLike = _as_runtime_type(TypeHints.Strings.StringLike)

    class Booleans:
        BooleanLike = _as_runtime_type(TypeHints.Booleans.BooleanLike)

    class Numerics:
        IntLike = _as_runtime_type(TypeHints.Numerics.IntLike)
        FloatLike = _as_runtime_type(TypeHints.Numerics.FloatLike)
        NumericLike = _as_runtime_type(TypeHints.Numerics.NumericLike)

    class Containers:
        SetLike = _as_runtime_type(TypeHints.Containers.SetLike)
        DictLike = _as_runtime_type(TypeHints.Containers.DictLike)
        ArrayLike = _as_runtime_type(TypeHints.Containers.ArrayLike)
        ContainerLike = _as_runtime_type(TypeHints.Containers.ContainerLike)

    class Sequences:
        ListLike = _as_runtime_type(TypeHints.Sequences.ListLike)
        TupleLike = _as_runtime_type(TypeHints.Sequences.TupleLike)
        SequenceLike = _as_runtime_type(TypeHints.Sequences.SequenceLike)


##
## === HELPER FUNCTIONS
##


def as_tuple(
    param,
    *,
    param_name: str = "<param>",
) -> tuple:
    """Return `param` as a tuple."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")
    if isinstance(param, tuple):
        return param
    if isinstance(param, list):
        return tuple(param)
    return (param, )


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
    param,
    *,
    valid_types: type | tuple[type, ...] | list[type],
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is an instance of the required type(s)."""
    if (param is None) and allow_none:
        return
    valid_types = _types_to_tuple(valid_types)
    if not isinstance(param, valid_types):
        valid_types_string = ", ".join(valid_type.__name__ for valid_type in valid_types)
        raise TypeError(
            f"`{param_name}` is {type(param).__name__}; expected one of: {valid_types_string}.",
        )


def ensure_not_none(
    param,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure a variable is not None."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")


##
## === INTERNAL HELPER
##


def _preview_indices(
    indices: list[int],
    *,
    preview_length: int = 5,
) -> str:
    """Return a short, human-readable preview of index positions."""
    if not indices:
        return "[]"
    preview = [str(index) for index in indices[:preview_length]]
    if len(indices) > preview_length:
        preview.append("...")
    return "[" + ", ".join(preview) + "]"


##
## === STRING LIKE
##


def ensure_string(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a string."""
    ensure_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Strings.StringLike,
    )


def ensure_nonempty_string(
    param: TypeHints.Strings.StringLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a non-empty string."""
    ensure_string(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(param) == 0:
        raise ValueError(f"`{param_name}` must be a non-empty string.")


def ensure_char(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a single-character string."""
    if (param is None) and allow_none:
        return
    ensure_string(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(param) != 1:
        raise ValueError(
            f"`{param_name}` must be a single-character string, got length {len(param)}.",
        )


##
## === BOOLEAN LIKE
##


def ensure_bool(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a boolean."""
    ensure_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Booleans.BooleanLike,
    )


def ensure_true(
    param,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value True."""
    ensure_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if not bool(param):
        raise ValueError(f"`{param_name}` must be True, got {param}.")


def ensure_false(
    param,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value False."""
    ensure_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if bool(param):
        raise ValueError(f"`{param_name}` must be False, got {param}.")


def ensure_not_bool(
    param,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is not a boolean."""
    if isinstance(param, RuntimeTypes.Booleans.BooleanLike):
        raise TypeError(f"`{param_name}` must not be a boolean.")


##
## === NUMERIC LIKE
##


def ensure_numeric(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a numeric (bools rejected)."""
    if (param is None) and allow_none:
        return
    ## reject booleans explicitly (they are subclasses of int)
    ensure_not_bool(
        param=param,
        param_name=param_name,
    )
    ensure_type(
        param=param,
        param_name=param_name,
        valid_types=RuntimeTypes.Numerics.NumericLike,
    )


def ensure_finite_numeric(
    param,
    *,
    param_name: str = "<param>",
    valid_types: tuple[type, ...],
    allow_none: bool,
    require_positive: bool,
    allow_zero: bool,
) -> None:
    """
    Ensure `param` is finite and of the given numeric types.

    If `require_positive` is:
      - False: no sign constraint.
      - True and `allow_zero` is True: require param >= 0.
      - True and `allow_zero` is False: require param > 0.
    """
    if param is None:
        if allow_none:
            return
        raise ValueError(f"`{param_name}` must not be None.")
    ## reject booleans explicitly (they are subclasses of int)
    ensure_not_bool(
        param=param,
        param_name=param_name,
    )
    if not isinstance(param, valid_types):
        valid_types_string = ", ".join(valid_type.__name__ for valid_type in valid_types)
        raise TypeError(
            f"`{param_name}` is {type(param).__name__}; expected: {valid_types_string}.",
        )
    if not numpy.isfinite(param):
        raise ValueError(f"`{param_name}` must be finite, got {param}.")
    if not(param >= 0) and (require_positive and allow_zero):
        raise ValueError(f"`{param_name}` must be non-negative (>= 0), got {param}.")
    if not(param > 0) and (require_positive and not allow_zero):
        raise ValueError(f"`{param_name}` must be positive (> 0), got {param}.")


def ensure_finite_float(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
    allow_zero: bool = True,
) -> None:
    """Ensure `param` is a finite float-like value."""
    ensure_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Numerics.FloatLike,
        require_positive=require_positive,
        allow_zero=allow_zero,
    )


def ensure_finite_int(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
    allow_zero: bool = True,
) -> None:
    """Ensure `param` is a finite int-like value."""
    ensure_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Numerics.IntLike,
        require_positive=require_positive,
        allow_zero=allow_zero,
    )


##
## === CONTAINER LIKE
##


def ensure_container(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is one of the supported container types."""
    ensure_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Containers.ContainerLike,
    )


##
## --- SEQUENCE LIKE
##


def ensure_sequence(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    seq_length: int | None = None,
    valid_seq_types: type | tuple[type, ...] | list[type] = RuntimeTypes.Sequences.SequenceLike,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
) -> None:
    """Ensure `param` is a valid sequence container, with optional fixed length and uniform element types."""
    if (param is None) and allow_none:
        return
    ## enforce container type
    valid_seq_types = _types_to_tuple(valid_seq_types)
    if not isinstance(param, valid_seq_types):
        valid_container_string = ", ".join(valid_container.__name__ for valid_container in valid_seq_types)
        raise TypeError(f"`{param_name}` must be one of ({valid_container_string}).")
    ## enforce number of elements
    if (seq_length is not None) and (len(param) != seq_length):
        raise ValueError(f"`{param_name}` must have length {seq_length}, got {len(param)}.")
    ## enforce uniform element types
    if valid_elem_types is not None:
        valid_elem_types = _types_to_tuple(valid_elem_types)
        bad_indices = [
            elem_index for elem_index, elem in enumerate(param) if not isinstance(elem, valid_elem_types)
        ]
        if bad_indices:
            valid_elem_types_string = ", ".join(valid_type.__name__ for valid_type in valid_elem_types)
            preview_bad_indices_string = _preview_indices(indices=bad_indices)
            raise TypeError(
                f"`{param_name}` elements must be of type(s) {valid_elem_types_string};"
                f" failed at indices: {preview_bad_indices_string}.",
            )


def ensure_nested_sequence(
    param,
    *,
    param_name: str = "<param>",
    outer_length: int | None = None,
    inner_length: int | None = None,
    valid_outer_types: type | tuple[type, ...] | list[type] = RuntimeTypes.Sequences.SequenceLike,
    valid_inner_types: type | tuple[type, ...] | list[type] = RuntimeTypes.Sequences.SequenceLike,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a nested (2D) sequence."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=outer_length,
        valid_seq_types=valid_outer_types,
        valid_elem_types=valid_inner_types,
    )
    for outer_index, inner_seq in enumerate(param):
        ensure_sequence(
            param=inner_seq,
            param_name=f"{param_name}[{outer_index}]",
            allow_none=False,
            seq_length=inner_length,
            valid_seq_types=valid_inner_types,
            valid_elem_types=valid_elem_types,
        )


##
## --- TUPLE LIKE
##


def ensure_flat_tuple(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a flat tuple (no nested containers)."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = (RuntimeTypes.Sequences.SequenceLike + RuntimeTypes.Containers.ContainerLike)
    bad_indices = [
        elem_index for elem_index, elem in enumerate(param) if isinstance(elem, invalid_elem_types)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` must be a flat tuple (no nested containers);"
            f" found nested container elements at indices: {preview_bad_indices_string}.",
        )


def ensure_nested_tuple(
    param,
    *,
    param_name: str = "<param>",
    outer_length: int | None = None,
    inner_length: int | None = None,
    valid_outer_types: type | tuple[type, ...] | list[type] = RuntimeTypes.Sequences.TupleLike,
    valid_inner_types: type | tuple[type, ...] | list[type] = RuntimeTypes.Sequences.TupleLike,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a nested (2D) tuple."""
    if (param is None) and allow_none:
        return
    ensure_nested_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        outer_length=outer_length,
        inner_length=inner_length,
        valid_outer_types=valid_outer_types,
        valid_inner_types=valid_inner_types,
        valid_elem_types=valid_elem_types,
    )


def ensure_tuple_of_strings(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of strings."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Strings.StringLike,
    )


def ensure_tuple_of_numbers(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of numbers (float or int, reject booleans)."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.NumericLike,
    )
    bad_indices = [
        elem_index for elem_index, elem in enumerate(param)
        if isinstance(elem, RuntimeTypes.Booleans.BooleanLike)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` elements must be numeric (int/float, not bool);"
            f" found booleans at indices: {preview_bad_indices_string}.",
        )


def ensure_tuple_of_floats(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of floats."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.FloatLike,
    )


def ensure_tuple_of_ints(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of integers."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.IntLike,
    )


def ensure_tuple_of_bools(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of booleans."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Booleans.BooleanLike,
    )


##
## --- LIST LIKE
##


def ensure_flat_list(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a flat list (no nested containers)."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = (RuntimeTypes.Sequences.SequenceLike + RuntimeTypes.Containers.ContainerLike)
    bad_indices = [
        elem_index for elem_index, elem in enumerate(param) if isinstance(elem, invalid_elem_types)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` must be a flat list (no nested containers);"
            f" found nested container elements at indices: {preview_bad_indices_string}.",
        )


def ensure_list_of_strings(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of strings."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Strings.StringLike,
    )


def ensure_list_of_numbers(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of numbers (float or int, booleans rejected)."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.NumericLike,
    )
    bad_indices = [
        elem_index for elem_index, elem in enumerate(param)
        if isinstance(elem, RuntimeTypes.Booleans.BooleanLike)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` elements must be numeric (int/float, not bool);"
            f" found boolean elements at indices: {preview_bad_indices_string}.",
        )


def ensure_list_of_floats(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of floats."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.FloatLike,
    )


def ensure_list_of_ints(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of integers."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.IntLike,
    )


def ensure_list_of_bools(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of booleans."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Booleans.BooleanLike,
    )


##
## --- DICT LIKE
##


def ensure_dict(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a dict."""
    ensure_type(
        param=param,
        allow_none=allow_none,
        param_name=param_name,
        valid_types=RuntimeTypes.Containers.DictLike,
    )


##
## --- ARRAY LIKE
##


def ensure_ndarray(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a NumPy array."""
    ensure_type(
        param=param,
        allow_none=allow_none,
        param_name=param_name,
        valid_types=RuntimeTypes.Containers.ArrayLike,
    )


def ensure_ndarray_ndim(
    param,
    *,
    ndim: int,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a NumPy array of fixed ndim."""
    if param is None:
        if allow_none:
            return
        raise ValueError(f"`{param_name}` must not be None.")
    ensure_ndarray(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if param.ndim != ndim:
        raise ValueError(f"`{param_name}` must have ndim={ndim}, got {param.ndim}.")


## } MODULE
