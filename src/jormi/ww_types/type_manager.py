## { MODULE
##
## === DEPENDENCIES
##

import numpy

from jormi.utils import list_utils

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
        valid_type_names = [valid_type.__name__ for valid_type in valid_types]
        valid_types_string = list_utils.as_string(
            elems=valid_type_names,
            wrap_in_quotes=True,
            conjunction="",
        )
        raise TypeError(f"`{param_name}` is {type(param).__name__}; expected {valid_types_string}.")


def ensure_not_none(
    param,
    *,
    param_name: str = "<param>",
) -> None:
    """Ensure a variable is not None."""
    if param is None:
        raise ValueError(f"`{param_name}` must not be None.")


##
## === STRING TYPES
##


class StringTypes:
    STRING = (str, )


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
        valid_types=StringTypes.STRING,
    )


def ensure_nonempty_string(
    param: str,
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
## === BOOLEAN TYPES
##


class BooleanTypes:
    BOOLEAN = (bool, numpy.bool_)


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
        valid_types=BooleanTypes.BOOLEAN,
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
    if isinstance(param, BooleanTypes.BOOLEAN):
        raise TypeError(f"`{param_name}` must not be a boolean.")


##
## === NUMERIC TYPES
##


class NumericTypes:
    INT = (int, numpy.integer)
    FLOAT = (float, numpy.floating)
    NUMERIC = INT + FLOAT


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
        valid_types=NumericTypes.NUMERIC,
        param_name=param_name,
    )


def ensure_finite_numeric(
    param,
    *,
    param_name: str = "<param>",
    valid_types: tuple[type, ...],
    allow_none: bool,
    require_positive: bool,
) -> None:
    """Ensure `param` is finite, positive (optional), and of the given numeric types."""
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
        valid_type_names = [valid_type.__name__ for valid_type in valid_types]
        valid_types_string = list_utils.as_string(
            elems=valid_type_names,
            wrap_in_quotes=True,
            conjunction="",
        )
        raise TypeError(f"`{param_name}` is {type(param).__name__}; expected: {valid_types_string}.")
    if not numpy.isfinite(param):
        raise ValueError(f"`{param_name}` must be finite, got {param}.")
    if require_positive and not (param > 0):
        raise ValueError(f"`{param_name}` must be positive (> 0), got {param}.")


def ensure_finite_float(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
) -> None:
    """Ensure `param` is a finite float-like value."""
    ensure_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=NumericTypes.FLOAT,
        require_positive=require_positive,
    )


def ensure_finite_int(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    require_positive: bool = False,
) -> None:
    """Ensure `param` is a finite int-like value."""
    ensure_finite_numeric(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        valid_types=NumericTypes.INT,
        require_positive=require_positive,
    )


##
## === CONTAINER
##


class ContainerTypes:
    SET = (set, )
    DICT = (dict, )
    ARRAY = (numpy.ndarray, )
    CONTAINER = SET + DICT + ARRAY


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
        valid_types=ContainerTypes.CONTAINER,
    )


##
## --- SEQUENCE
##


class SequenceTypes(ContainerTypes):
    LIST = (list, )
    TUPLE = (tuple, )
    SEQUENCE = LIST + TUPLE


def ensure_sequence(
    param,
    *,
    param_name: str = "<param>",
    allow_none: bool = False,
    seq_length: int | None = None,
    valid_seq_types: type | tuple[type, ...] | list[type] = SequenceTypes.SEQUENCE,
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
            preview_bad_indices_string = list_utils.get_preview_string(
                elems=bad_indices,
                preview_length=5,
            )
            valid_type_names = [valid_elem_type.__name__ for valid_elem_type in valid_elem_types]
            valid_elem_types_string = list_utils.as_string(
                elems=valid_type_names,
                wrap_in_quotes=True,
                conjunction="",
            )
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
## --- TUPLE
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
        valid_seq_types=SequenceTypes.TUPLE,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = SequenceTypes.SEQUENCE + ContainerTypes.CONTAINER
    bad_indices: list[int] = []
    for elem_index, elem in enumerate(param):
        if isinstance(elem, invalid_elem_types):
            bad_indices.append(elem_index)
    if bad_indices:
        preview_bad_indices_string = list_utils.get_preview_string(
            elems=bad_indices,
            preview_length=5,
        )
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
    valid_outer_types: type | tuple[type, ...] | list[type] = SequenceTypes.TUPLE,
    valid_inner_types: type | tuple[type, ...] | list[type] = SequenceTypes.TUPLE,
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


def ensure_tuple_of_numbers(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a tuple of numeric scalars."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=SequenceTypes.TUPLE,
        valid_elem_types=NumericTypes.NUMERIC,
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
        valid_seq_types=SequenceTypes.TUPLE,
        valid_elem_types=BooleanTypes.BOOLEAN,
    )


##
## --- LIST
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
        valid_seq_types=SequenceTypes.LIST,
        valid_elem_types=valid_elem_types,
    )
    invalid_elem_types = SequenceTypes.SEQUENCE + ContainerTypes.CONTAINER
    bad_indices: list[int] = []
    for elem_index, elem in enumerate(param):
        if isinstance(elem, invalid_elem_types):
            bad_indices.append(elem_index)
    if bad_indices:
        preview_bad_indices_string = list_utils.get_preview_string(
            elems=bad_indices,
            preview_length=5,
        )
        raise TypeError(
            f"`{param_name}` must be a flat list (no nested containers);"
            f" found nested container elements at indices: {preview_bad_indices_string}.",
        )


def ensure_list_of_numbers(
    param,
    *,
    param_name: str = "<param>",
    seq_length: int | None = None,
    allow_none: bool = False,
) -> None:
    """Ensure `param` is a list of numeric scalars."""
    ensure_sequence(
        param=param,
        param_name=param_name,
        allow_none=allow_none,
        seq_length=seq_length,
        valid_seq_types=SequenceTypes.LIST,
        valid_elem_types=NumericTypes.NUMERIC,
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
        valid_seq_types=SequenceTypes.LIST,
        valid_elem_types=StringTypes.STRING,
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
        valid_seq_types=SequenceTypes.LIST,
        valid_elem_types=BooleanTypes.BOOLEAN,
    )


##
## --- DICT
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
        valid_types=ContainerTypes.DICT,
        param_name=param_name,
    )


##
## --- ARRAY
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
        valid_types=ContainerTypes.ARRAY,
        param_name=param_name,
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
