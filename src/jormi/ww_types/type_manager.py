## { MODULE
##
## === DEPENDENCIES
##

import numpy

from jormi.utils import list_utils

##
## === TYPES
##


class StringTypes:
    STRING = (str, )


class BooleanTypes:
    BOOLEAN = (bool, numpy.bool_)


class NumericTypes:
    INT = (int, numpy.integer)
    FLOAT = (float, numpy.floating)
    NUMERIC = INT + FLOAT


class ContainerTypes:
    SET = (set, )
    DICT = (dict, )
    ARRAY = (numpy.ndarray, )
    CONTAINER = SET + DICT + ARRAY


class SequenceTypes(ContainerTypes):
    LIST = (list, )
    TUPLE = (tuple, )
    SEQUENCE = LIST + TUPLE


##
## === FUNCTIONS
##


def as_tuple(
    *,
    param,
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
    *,
    param,
    valid_types: type | tuple[type, ...] | list[type],
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is an instance of the required type(s)."""
    if (param is None) and allow_none:
        return
    valid_types = _types_to_tuple(valid_types)
    if not isinstance(param, valid_types):
        valid_type_names = [valid_type.__name__ for valid_type in valid_types]
        valid_types_str = list_utils.cast_to_string(
            elems=valid_type_names,
            wrap_in_quotes=True,
            conjunction="",
        )
        raise TypeError(f"`{param_name}` is {type(param).__name__}; expected {valid_types_str}.")


def ensure_not_none(
    *,
    param,
    param_name: str = "<param>",
) -> None:
    """Ensure a variable is not None."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")


def ensure_char(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a single-character string."""
    if (param is None) and allow_none:
        return
    ensure_str(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(param) != 1:
        raise ValueError(
            f"`{param_name}` must be a single-character string (got length {len(param)}).",
        )


def ensure_str(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a string."""
    ensure_type(
        param=param,
        valid_types=StringTypes.STRING,
        param_name=param_name,
        allow_none=allow_none,
    )


def ensure_nonempty_str(
    *,
    param: str,
    param_name: str = "<param>",
) -> None:
    """Enforce a non-empty string."""
    ensure_str(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if len(param) == 0:
        raise ValueError(f"`{param_name}` must be a non-empty string.")


def ensure_bool(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a boolean."""
    ensure_type(
        param=param,
        valid_types=BooleanTypes.BOOLEAN,
        param_name=param_name,
        allow_none=allow_none,
    )


def ensure_true(
    *,
    param,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value True."""
    ensure_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if not bool(param):
        raise ValueError(f"`{param_name}` must be True (got {param}).")


def ensure_false(
    *,
    param,
    param_name: str = "<param>",
) -> None:
    """Ensure `param` is a boolean with value False."""
    ensure_bool(
        param=param,
        param_name=param_name,
        allow_none=False,
    )
    if bool(param):
        raise ValueError(f"`{param_name}` must be False (got {param}).")


def ensure_not_boolean(
    *,
    param,
    param_name: str = "<param>",
) -> None:
    if isinstance(param, BooleanTypes.BOOLEAN):
        raise TypeError(f"`{param_name}` must not be a boolean.")


def ensure_numeric(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a numeric (bools rejected)."""
    if (param is None) and allow_none:
        return
    ## reject booleans explicitly (they are subclasses of int)
    ensure_not_boolean(
        param=param,
        param_name=param_name,
    )
    ensure_type(
        param=param,
        valid_types=NumericTypes.NUMERIC,
        param_name=param_name,
    )


def ensure_finite_number(
    *,
    param,
    param_name: str,
    valid_types: tuple[type, ...],
    allow_none: bool,
) -> None:
    """Internal helper: ensure `param` is finite and of the given numeric types."""
    if param is None:
        if allow_none:
            return
        raise ValueError(f"`{param_name}` must not be None.")
    ## reject booleans explicitly (they are subclasses of int)
    ensure_not_boolean(
        param=param,
        param_name=param_name,
    )
    if not isinstance(param, valid_types):
        valid_type_names = [valid_type.__name__ for valid_type in valid_types]
        valid_types_str = list_utils.cast_to_string(
            elems=valid_type_names,
            wrap_in_quotes=True,
            conjunction="",
        )
        raise TypeError(f"`{param_name}` is {type(param).__name__}; expected: {valid_types_str}.")
    if not numpy.isfinite(param):
        raise ValueError(f"`{param_name}` must be finite (got {param}).")


def ensure_finite_float(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a finite float-like value."""
    ensure_finite_number(
        param=param,
        param_name=param_name,
        valid_types=NumericTypes.FLOAT,
        allow_none=allow_none,
    )


def ensure_finite_int(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a finite int-like value."""
    ensure_finite_number(
        param=param,
        param_name=param_name,
        valid_types=NumericTypes.INT,
        allow_none=allow_none,
    )


def ensure_sequence(
    *,
    param,
    param_name: str = "<seq>",
    seq_length: int | None = None,
    valid_seq_types: type | tuple[type, ...] | list[type] = SequenceTypes.SEQUENCE,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a valid sequence container, with optional fixed length and uniform element types."""
    if (param is None) and allow_none:
        return
    ## enforce container type
    valid_seq_types = _types_to_tuple(valid_seq_types)
    if not isinstance(param, valid_seq_types):
        valid_container_str = ", ".join(valid_container.__name__ for valid_container in valid_seq_types)
        raise TypeError(f"`{param_name}` must be one of ({valid_container_str}).")
    ## enforce number of elements
    if (seq_length is not None) and (len(param) != seq_length):
        raise ValueError(f"`{param_name}` must have length {seq_length} (got {len(param)}).")
    ## enforce uniform element types
    if valid_elem_types is not None:
        valid_elem_types = _types_to_tuple(valid_elem_types)
        bad_indices = [
            elem_index for elem_index, elem in enumerate(param) if not isinstance(elem, valid_elem_types)
        ]
        if bad_indices:
            preview_bad_indices_str = list_utils.get_preview_string(
                elems=bad_indices,
                preview_length=5,
            )
            valid_type_names = [valid_elem_type.__name__ for valid_elem_type in valid_elem_types]
            valid_elem_types_str = list_utils.cast_to_string(
                elems=valid_type_names,
                wrap_in_quotes=True,
                conjunction="",
            )
            raise TypeError(
                f"`{param_name}` elements must be of type(s) {valid_elem_types_str}; "
                f"failed at indices: {preview_bad_indices_str}.",
            )


def ensure_nested_sequence(
    *,
    param,
    param_name: str = "<seq>",
    outer_length: int | None = None,
    inner_length: int | None = None,
    valid_outer_types: type | tuple[type, ...] | list[type] = SequenceTypes.SEQUENCE,
    valid_inner_types: type | tuple[type, ...] | list[type] = SequenceTypes.SEQUENCE,
    valid_elem_types: type | tuple[type, ...] | list[type] | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a nested (2D) sequence."""
    if (param is None) and allow_none:
        return
    ensure_sequence(
        param=param,
        param_name=param_name,
        seq_length=outer_length,
        valid_seq_types=valid_outer_types,
        valid_elem_types=valid_inner_types,
        allow_none=False,
    )
    for outer_index, inner_seq in enumerate(param):
        ensure_sequence(
            param=inner_seq,
            param_name=f"{param_name}[{outer_index}]",
            seq_length=inner_length,
            valid_seq_types=valid_inner_types,
            valid_elem_types=valid_elem_types,
            allow_none=False,
        )


def ensure_container(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is one of the supported container types."""
    ensure_type(
        param=param,
        valid_types=ContainerTypes.CONTAINER,
        param_name=param_name,
        allow_none=allow_none,
    )


def ensure_dict(
    *,
    param,
    param_name: str = "<param>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a dict."""
    ensure_type(
        param=param,
        valid_types=ContainerTypes.DICT,
        param_name=param_name,
        allow_none=allow_none,
    )


def ensure_ndarray(
    *,
    param,
    param_name: str = "<array>",
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a NumPy array."""
    ensure_type(
        param=param,
        valid_types=ContainerTypes.ARRAY,
        param_name=param_name,
        allow_none=allow_none,
    )


def ensure_ndarray_ndim(
    *,
    param,
    ndim: int,
    param_name: str = "<array>",
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
        raise ValueError(f"`{param_name}` must have ndim={ndim} (got {param.ndim}).")


def ensure_numeric_sequence(
    *,
    param,
    param_name: str = "<seq>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a (list/tuple) of numeric scalars."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        valid_seq_types=SequenceTypes.SEQUENCE,
        seq_length=seq_length,
        valid_elem_types=NumericTypes.NUMERIC,
        allow_none=allow_none,
    )


def ensure_str_sequence(
    *,
    param,
    param_name: str = "<seq>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a (list/tuple) of strings."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        valid_seq_types=SequenceTypes.SEQUENCE,
        seq_length=seq_length,
        valid_elem_types=StringTypes.STRING,
        allow_none=allow_none,
    )


def ensure_bool_sequence(
    *,
    param,
    param_name: str = "<seq>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a (list/tuple) of booleans."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        valid_seq_types=SequenceTypes.SEQUENCE,
        seq_length=seq_length,
        valid_elem_types=BooleanTypes.BOOLEAN,
        allow_none=allow_none,
    )


## } MODULE
