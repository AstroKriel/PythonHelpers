## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, cast, get_args

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_types import python_types

##
## === RUNTIME TYPES
## derived from `python_types.Types` for isinstance checks inside `validate_*` functions.
##


def _as_runtime_type(
    type_hint: Any,
) -> tuple[type, ...]:
    """
    Convert a type-hint alias (possibly a union) into a tuple of runtime
    types suitable for isinstance checks.
    """
    args = get_args(type_hint)
    if args:
        if not all(isinstance(arg, type) for arg in args):
            raise TypeError(f"non-type argument(s) in hint: {type_hint!r}.")
        return tuple(args)
    if isinstance(type_hint, type):
        return (type_hint, )
    raise TypeError(f"unsupported type-hint: {type_hint!r}.")


class RuntimeTypes:
    """Runtime type tuples derived from `python_types.Types` for isinstance checks."""

    class Strings:
        StringLike = _as_runtime_type(python_types.Types.Strings.StringLike)

    class Booleans:
        BooleanLike = _as_runtime_type(python_types.Types.Booleans.BooleanLike)

    class Numerics:
        IntLike = _as_runtime_type(python_types.Types.Numerics.IntLike)
        FloatLike = _as_runtime_type(python_types.Types.Numerics.FloatLike)
        NumericLike = _as_runtime_type(python_types.Types.Numerics.NumericLike)

    class Containers:
        SetLike = _as_runtime_type(python_types.Types.Containers.SetLike)
        DictLike = _as_runtime_type(python_types.Types.Containers.DictLike)
        ArrayLike = _as_runtime_type(python_types.Types.Containers.ArrayLike)
        ContainerLike = _as_runtime_type(python_types.Types.Containers.ContainerLike)

    class Sequences:
        ListLike = _as_runtime_type(python_types.Types.Sequences.ListLike)
        TupleLike = _as_runtime_type(python_types.Types.Sequences.TupleLike)
        SequenceLike = _as_runtime_type(python_types.Types.Sequences.SequenceLike)


##
## === HELPER FUNCTIONS
##


def as_tuple(
    param: object,
    *,
    param_name: str = "<param>",
) -> tuple[Any, ...]:
    """Return `param` as a tuple."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")
    if isinstance(param, tuple):
        return cast(tuple[Any, ...], param)
    if isinstance(param, list):
        return tuple(cast(list[Any], param))
    return (param, )


def _types_to_tuple(
    valid_types: type | tuple[type, ...] | list[type],
) -> tuple[type, ...]:
    """Canonicalize a type or sequence of types to a (type, ...) tuple."""
    if isinstance(valid_types, type):
        return (valid_types, )
    if isinstance(valid_types, (tuple, list)):  # pyright: ignore[reportUnnecessaryIsInstance]
        if not valid_types:
            raise ValueError("empty type specification.")
        if not all(isinstance(valid_type, type)  # pyright: ignore[reportUnnecessaryIsInstance]
                   for valid_type in valid_types):
            raise TypeError("`valid_types` entries must be valid Python types.")
        return tuple(valid_types)
    raise TypeError(
        "`valid_types` must be a type or a tuple/list of types.",
    )  # pyright: ignore[reportUnreachable]


def validate_type(
    param: object,
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


def validate_not_none(
    param: object,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure a variable is not None."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")


##
## === INTERNAL HELPERS
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


def validate_string(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a string."""
    validate_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Strings.StringLike,
    )


def validate_nonempty_string(
    param: python_types.Types.Strings.StringLike,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a non-empty string."""
    validate_string(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(param) == 0:
        raise ValueError(f"`{param_name}` must be a non-empty string.")


def validate_char(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a single-character string."""
    if (param is None) and allow_none:
        return
    validate_string(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(cast(str, param)) != 1:
        raise ValueError(
            f"`{param_name}` must be a single-character string, got length {len(cast(str, param))}.",
        )


##
## === BOOLEAN LIKE
##


def validate_bool(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a boolean."""
    validate_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Booleans.BooleanLike,
    )


def validate_true(
    param: object,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value True."""
    validate_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if not bool(param):
        raise ValueError(f"`{param_name}` must be True, got {param}.")


def validate_false(
    param: object,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value False."""
    validate_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if bool(param):
        raise ValueError(f"`{param_name}` must be False, got {param}.")


def validate_not_bool(
    param: object,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is not a boolean."""
    if isinstance(param, RuntimeTypes.Booleans.BooleanLike):
        raise TypeError(f"`{param_name}` must not be a boolean.")


##
## === NUMERIC LIKE
##


def validate_numeric(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a numeric (bools rejected)."""
    if (param is None) and allow_none:
        return
    ## reject booleans explicitly (they are subclasses of int)
    validate_not_bool(
        param=param,
        param_name=param_name,
    )
    validate_type(
        param=param,
        param_name=param_name,
        valid_types=RuntimeTypes.Numerics.NumericLike,
    )


def validate_finite_numeric(
    param: object,
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
    validate_not_bool(
        param=param,
        param_name=param_name,
    )
    if not isinstance(param, valid_types):
        valid_types_string = ", ".join(valid_type.__name__ for valid_type in valid_types)
        raise TypeError(
            f"`{param_name}` is {type(param).__name__}; expected: {valid_types_string}.",
        )
    _numeric = cast(float | int, param)
    if not numpy.isfinite(_numeric):
        raise ValueError(f"`{param_name}` must be finite, got {param}.")
    if not (_numeric >= 0) and (require_positive and allow_zero):
        raise ValueError(f"`{param_name}` must be non-negative (>= 0), got {param}.")
    if not (_numeric > 0) and (require_positive and not allow_zero):
        raise ValueError(f"`{param_name}` must be positive (> 0), got {param}.")


def validate_finite_float(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
    allow_zero: bool = True,
) -> None:
    """Ensure `param` is a finite float-like value."""
    validate_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Numerics.FloatLike,
        require_positive=require_positive,
        allow_zero=allow_zero,
    )


def validate_finite_int(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
    allow_zero: bool = True,
) -> None:
    """Ensure `param` is a finite int-like value."""
    validate_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Numerics.IntLike,
        require_positive=require_positive,
        allow_zero=allow_zero,
    )


def validate_finite_scalar(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
    allow_zero: bool = True,
) -> None:
    """Ensure `param` is a finite scalar (int-like or float-like) value."""
    validate_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Numerics.NumericLike,
        require_positive=require_positive,
        allow_zero=allow_zero,
    )


def validate_in_bounds(
    param: object,
    *,
    min_value: float,
    max_value: float,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a finite scalar in [min_value, max_value]."""
    if (param is None) and allow_none:
        return
    validate_finite_scalar(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    _param_f = cast(float, param)
    if not (float(min_value) <= float(_param_f) <= float(max_value)):
        raise ValueError(
            f"`{param_name}` must lie in [{min_value}, {max_value}], got {param}.",
        )


##
## === CONTAINER LIKE
##


def validate_container(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is one of the supported container types."""
    validate_type(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=RuntimeTypes.Containers.ContainerLike,
    )


##
## --- SEQUENCE LIKE
##


def validate_sequence(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    seq_length: int | None = None,
    valid_seq_types: type
    | tuple[type, ...]
    | list[type] = RuntimeTypes.Sequences.SequenceLike,
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
    _param_seq = cast(list[Any] | tuple[Any, ...], param)
    ## enforce number of elements
    if (seq_length is not None) and (len(_param_seq) != seq_length):
        raise ValueError(
            f"`{param_name}` must have length {seq_length}, got {len(_param_seq)}.",
        )
    ## enforce uniform element types
    if valid_elem_types is not None:
        valid_elem_types = _types_to_tuple(valid_elem_types)
        bad_indices = [
            elem_index for elem_index, elem in enumerate(_param_seq)
            if not isinstance(elem, valid_elem_types)
        ]
        if bad_indices:
            valid_elem_types_string = ", ".join(valid_type.__name__ for valid_type in valid_elem_types)
            preview_bad_indices_string = _preview_indices(indices=bad_indices)
            raise TypeError(
                f"`{param_name}` elements must be of type(s) {valid_elem_types_string};"
                f" failed at indices: {preview_bad_indices_string}.",
            )


def validate_nested_sequence(
    param: object,
    *,
    param_name: str = "<param>",
    outer_length: int | None = None,
    inner_length: int | None = None,
    valid_outer_types: type
    | tuple[type, ...]
    | list[type] = RuntimeTypes.Sequences.SequenceLike,
    valid_inner_types: type
    | tuple[type, ...]
    | list[type] = RuntimeTypes.Sequences.SequenceLike,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a nested (2D) sequence."""
    if (param is None) and allow_none:
        return
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=outer_length,
        valid_seq_types=valid_outer_types,
        valid_elem_types=valid_inner_types,
    )
    for outer_index, inner_seq in enumerate(cast(list[Any] | tuple[Any, ...], param)):
        validate_sequence(
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


def validate_flat_tuple(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a flat tuple (no nested containers)."""
    if (param is None) and allow_none:
        return
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = (RuntimeTypes.Sequences.SequenceLike + RuntimeTypes.Containers.ContainerLike)
    bad_indices = [
        elem_index for elem_index, elem in enumerate(cast(tuple[Any, ...], param))
        if isinstance(elem, invalid_elem_types)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` must be a flat tuple (no nested containers);"
            f" found nested container elements at indices: {preview_bad_indices_string}.",
        )


def validate_nested_tuple(
    param: object,
    *,
    param_name: str = "<param>",
    outer_length: int | None = None,
    inner_length: int | None = None,
    valid_outer_types: type
    | tuple[type, ...]
    | list[type] = RuntimeTypes.Sequences.TupleLike,
    valid_inner_types: type
    | tuple[type, ...]
    | list[type] = RuntimeTypes.Sequences.TupleLike,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a nested (2D) tuple."""
    if (param is None) and allow_none:
        return
    validate_nested_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        outer_length=outer_length,
        inner_length=inner_length,
        valid_outer_types=valid_outer_types,
        valid_inner_types=valid_inner_types,
        valid_elem_types=valid_elem_types,
    )


def validate_tuple_of_strings(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of strings."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Strings.StringLike,
    )


def validate_tuple_of_numbers(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of numbers (float or int, reject booleans)."""
    if (param is None) and allow_none:
        return
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.NumericLike,
    )
    bad_indices = [
        elem_index for elem_index, elem in enumerate(cast(tuple[Any, ...], param))
        if isinstance(elem, RuntimeTypes.Booleans.BooleanLike)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` elements must be numeric (int/float, not bool);"
            f" found booleans at indices: {preview_bad_indices_string}.",
        )


def validate_ordered_pair(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    strict_ordering: bool = False,
) -> None:
    """Ensure `param` is a 2-element numeric tuple where param[0] <= param[1].
    If strict_ordering=True, enforce param[0] < param[1] (equal values not allowed).
    """
    if (param is None) and allow_none:
        return
    validate_tuple_of_numbers(
        param=param,
        param_name=param_name,
        seq_length=2,
        allow_none=False,
    )
    _param_t = cast(tuple[Any, ...], param)
    min_value, max_value = float(_param_t[0]), float(_param_t[1])
    if strict_ordering:
        if not (min_value < max_value):
            raise ValueError(
                f"`{param_name}` must satisfy [0] < [1] (strict), got {param}.",
            )
    else:
        if not (min_value <= max_value):
            raise ValueError(f"`{param_name}` must satisfy [0] <= [1], got {param}.")


def validate_tuple_of_floats(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of floats."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.FloatLike,
    )


def validate_tuple_of_ints(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of integers."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.TupleLike,
        valid_elem_types=RuntimeTypes.Numerics.IntLike,
    )


def validate_tuple_of_bools(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of booleans."""
    validate_sequence(
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


def validate_flat_list(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a flat list (no nested containers)."""
    if (param is None) and allow_none:
        return
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = (RuntimeTypes.Sequences.SequenceLike + RuntimeTypes.Containers.ContainerLike)
    bad_indices = [
        elem_index for elem_index, elem in enumerate(cast(list[Any], param))
        if isinstance(elem, invalid_elem_types)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` must be a flat list (no nested containers);"
            f" found nested container elements at indices: {preview_bad_indices_string}.",
        )


def validate_list_of_strings(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of strings."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Strings.StringLike,
    )


def validate_list_of_numbers(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of numbers (float or int, booleans rejected)."""
    if (param is None) and allow_none:
        return
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=False,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.NumericLike,
    )
    bad_indices = [
        elem_index for elem_index, elem in enumerate(cast(list[Any], param))
        if isinstance(elem, RuntimeTypes.Booleans.BooleanLike)
    ]
    if bad_indices:
        preview_bad_indices_string = _preview_indices(indices=bad_indices)
        raise TypeError(
            f"`{param_name}` elements must be numeric (int/float, not bool);"
            f" found boolean elements at indices: {preview_bad_indices_string}.",
        )


def validate_list_of_floats(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of floats."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.FloatLike,
    )


def validate_list_of_ints(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of integers."""
    validate_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=RuntimeTypes.Sequences.ListLike,
        valid_elem_types=RuntimeTypes.Numerics.IntLike,
    )


def validate_list_of_bools(
    param: object,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of booleans."""
    validate_sequence(
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


def validate_dict(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a dict."""
    validate_type(
        param=param,
        allow_none=allow_none,
        param_name=param_name,
        valid_types=RuntimeTypes.Containers.DictLike,
    )


##
## --- ARRAY LIKE
##


def validate_ndarray(
    param: object,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a NumPy array."""
    validate_type(
        param=param,
        allow_none=allow_none,
        param_name=param_name,
        valid_types=RuntimeTypes.Containers.ArrayLike,
    )


def validate_ndarray_ndim(
    param: object,
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
    validate_ndarray(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if cast(NDArray[Any], param).ndim != ndim:
        raise ValueError(
            f"`{param_name}` must have ndim={ndim}, got {cast(NDArray[Any], param).ndim}.",
        )


## } MODULE
