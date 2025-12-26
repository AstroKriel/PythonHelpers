## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import Literal

from jormi.ww_types import type_checks

##
## === FUNCTIONS
##


def sample_list(
    *,
    elems: list,
    max_elems: int,
) -> list:
    """Return at most `max_elems` samples from `elems`, spread across the list."""
    type_checks.ensure_sequence(
        param=elems,
        param_name="elems",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_finite_int(
        param=max_elems,
        param_name="max_elems",
        allow_none=False,
        require_positive=True,
    )
    num_elems = len(elems)
    if num_elems == 0:
        raise ValueError("`elems` must be non-empty.")
    if max_elems == 1:
        return [elems[0]]
    if num_elems <= max_elems:
        return elems
    index_stride = (num_elems - 1) // (max_elems - 1)
    indices_to_keep = [round(index * index_stride) for index in range(max_elems)]
    return [elems[elem_index] for elem_index in indices_to_keep]


def filter_out_nones(
    elems: list,
) -> list:
    """Return a copy of `elems` with all `None` entries removed."""
    type_checks.ensure_sequence(
        param=elems,
        param_name="elems",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    return [elem for elem in elems if elem is not None]


def get_index_of_closest_value(
    *,
    values: list,
    target: float,
) -> int:
    """Find the index of the closest value to a `target` value in a list."""
    type_checks.ensure_sequence(
        param=values,
        param_name="values",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
        valid_elem_types=type_checks.RuntimeTypes.Numerics.NumericLike,
    )
    type_checks.ensure_numeric(
        param=target,
        param_name="target",
        allow_none=False,
    )
    if len(values) == 0:
        raise ValueError("`values` must be non-empty.")
    values_array = numpy.asarray(values)
    if target == numpy.inf:
        return int(numpy.nanargmax(values_array))
    if target == -numpy.inf:
        return int(numpy.nanargmin(values_array))
    return int(numpy.nanargmin(numpy.abs(values_array - target)))


def get_index_of_first_crossing(
    *,
    values: list[float],
    target: float,
    direction: Literal["rising", "falling"] | None = None,
) -> int | numpy.integer | None:
    """
    Return the index of the first interval where the list crosses `target`.

    Parameters
    ----------
    values:
        1D list of numeric values.
    target:
        Threshold value to detect crossings of.
    direction:
        "rising"  -> only crossings where value increases through target
        "falling" -> only crossings where value decreases through target
        None      -> either direction
    """
    type_checks.ensure_sequence(
        param=values,
        param_name="values",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
        valid_elem_types=type_checks.RuntimeTypes.Numerics.NumericLike,
    )
    type_checks.ensure_numeric(
        param=target,
        param_name="target",
        allow_none=False,
    )
    values_array = numpy.asarray(values)
    if values_array.size == 0:
        raise ValueError("`values` must be non-empty.")
    min_value = numpy.min(values_array)
    max_value = numpy.max(values_array)
    if not (min_value <= target <= max_value):
        raise ValueError(
            f"`target` ({target:.2f}) is outside the range of the input values:"
            f" [{min_value:.2f}, {max_value:.2f}].",
        )
    valid_directions = ["rising", "falling", None]
    if direction not in valid_directions:
        valid_string = as_quoted_string(valid_directions)
        raise ValueError(
            f"`direction` must be one of {valid_string}, got {direction!r}.",
        )
    ## handle endpoints exactly
    if target == min_value:
        return numpy.argmin(values_array)
    if target == max_value:
        return numpy.argmax(values_array)
    ## scan for first crossing interval
    for value_index in range(len(values_array) - 1):
        value_left = values_array[value_index]
        value_right = values_array[value_index + 1]
        crossed_rising = (value_left < target <= value_right)
        crossed_falling = (value_right < target <= value_left)
        if (direction == "rising") and crossed_rising:
            return value_index
        if (direction == "falling") and crossed_falling:
            return value_index
        if (direction is None) and (crossed_rising or crossed_falling):
            return value_index
    return None


def as_string(
    elems: list,
    *,
    wrap_in_quotes: bool = False,
    conjunction: str = "",
) -> str:
    """Convert a (possibly nested) list into a human-readable, comma-separated string."""
    type_checks.ensure_sequence(
        param=elems,
        param_name="elems",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_string(
        param=conjunction,
        param_name="conjunction",
        allow_none=False,
    )
    type_checks.ensure_bool(
        param=wrap_in_quotes,
        param_name="wrap_in_quotes",
        allow_none=False,
    )
    elems = flatten_list(elems)
    if len(elems) == 0:
        return ""
    elems = [f"`{elem}`" if wrap_in_quotes else str(elem) for elem in elems]
    conjunction = conjunction.strip()
    if conjunction == "":
        return ", ".join(elems)
    if len(elems) == 2:
        return f"{elems[0]} {conjunction} {elems[1]}"
    return ", ".join(elems[:-1]) + f" {conjunction} {elems[-1]}"


def as_quoted_string(
    elems: list,
) -> str:
    """Return a comma-separated string of elements wrapped in backticks."""
    return as_string(
        elems=elems,
        wrap_in_quotes=True,
        conjunction="",
    )


def get_preview_string(
    elems: list,
    *,
    preview_length: int = 5,
    wrap_in_quotes: bool = False,
) -> str:
    """Return a short preview string of the first few elements."""
    type_checks.ensure_sequence(
        param=elems,
        param_name="elems",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_finite_int(
        param=preview_length,
        param_name="preview_length",
        allow_none=False,
        require_positive=True,
    )
    type_checks.ensure_bool(
        param=wrap_in_quotes,
        param_name="wrap_in_quotes",
        allow_none=False,
    )
    elems_preview = as_string(
        elems=elems[:preview_length],
        wrap_in_quotes=wrap_in_quotes,
        conjunction="",
    )
    if len(elems) > preview_length:
        return elems_preview + "..."
    return elems_preview


def get_intersect_of_lists(
    *,
    list_a: list,
    list_b: list,
    sort_values: bool = False,
) -> list:
    """Find the intersection of two lists (optionally sorted)."""
    type_checks.ensure_sequence(
        param=list_a,
        param_name="list_a",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_sequence(
        param=list_b,
        param_name="list_b",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_bool(
        param=sort_values,
        param_name="sort_values",
        allow_none=False,
    )
    if (len(list_a) == 0) or (len(list_b) == 0):
        return []
    set_intersect = set(list_a) & set(list_b)
    if sort_values:
        return sorted(set_intersect)
    return list(set_intersect)


def get_union_of_lists(
    *,
    list_a: list,
    list_b: list,
    sort_values: bool = False,
) -> list:
    """Find the union of two lists (optionally sorted)."""
    type_checks.ensure_sequence(
        param=list_a,
        param_name="list_a",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_sequence(
        param=list_b,
        param_name="list_b",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    type_checks.ensure_bool(
        param=sort_values,
        param_name="sort_values",
        allow_none=False,
    )
    if (len(list_a) == 0) or (len(list_b) == 0):
        return list(list_a) + list(list_b)
    set_union = set(list_a) | set(list_b)
    if sort_values:
        return sorted(set_union)
    return list(set_union)


def flatten_list(
    elems: list,
) -> list:
    """Flatten a nested list into a single list."""
    type_checks.ensure_sequence(
        param=elems,
        param_name="elems",
        valid_seq_types=type_checks.RuntimeTypes.Sequences.ListLike,
    )
    flat_elems: list = []
    for elem in elems:
        if isinstance(elem, list):
            flat_elems.extend(flatten_list(elem))
        else:
            flat_elems.append(elem)
    return flat_elems


## } MODULE
