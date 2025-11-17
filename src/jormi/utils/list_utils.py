## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_manager

##
## === FUNCTIONS
##


def sample_list(
    elems: list,
    max_elems: int,
) -> list:
    num_elems = len(elems)
    if num_elems == 0: raise ValueError("`elems` must be non-empty.")
    if max_elems < 1: raise ValueError("`max_elems` must be >= 1.")
    if max_elems == 1: return [elems[0]]
    if num_elems <= max_elems: return elems
    index_stride = (num_elems - 1) // (max_elems - 1)
    indices_to_keep = [round(_index * index_stride) for _index in range(max_elems)]
    return [elems[elem_index] for elem_index in indices_to_keep]


def filter_out_nones(
    elems: list,
) -> list:
    return [elem for elem in elems if elem is not None]


def get_index_of_closest_value(
    values: list,
    target: float,
) -> int:
    """Find the index of the closest value to a `target` value."""
    type_manager.ensure_type(
        param=values,
        valid_types=(list, numpy.ndarray),
    )
    type_manager.ensure_type(
        param=target,
        valid_types=(int, float),
    )
    if len(values) == 0: raise ValueError("Input list cannot be empty")
    array = numpy.asarray(values)
    if target is None: return None
    if target == numpy.inf: return int(numpy.nanargmax(array))
    if target == -numpy.inf: return int(numpy.nanargmin(array))
    return int(numpy.nanargmin(numpy.abs(array - target)))


def get_index_of_first_crossing(
    values: list[float],
    target: float,
    direction: str | None = None,
) -> None:
    values = numpy.asarray(values)
    min_value = numpy.min(values)
    max_value = numpy.max(values)
    if not (min_value <= target <= max_value):
        raise ValueError(
            f"`target` ({target:.2f}) is outside the range of the input values: [{min_value:.2f}, {max_value:.2f}].",
        )
    valid_filters = ["rising", "falling", None]
    if direction not in valid_filters:
        raise ValueError(
            f"`direction` must be one of {valid_filters}, but got {direction!r}. Choose from {cast_to_string(valid_filters)}",
        )
    if target == min_value:
        return numpy.argmin(values)
    if target == max_value:
        return numpy.argmax(values)
    for value_index in range(len(values) - 1):
        value_left = values[value_index]
        value_right = values[value_index + 1]
        crossed_target_while_rising = (value_left < target <= value_right)
        crossed_target_while_falling = (value_right < target <= value_left)
        if (direction == "rising") and crossed_target_while_rising:
            return value_index
        elif (direction == "falling") and crossed_target_while_falling:
            return value_index
        elif (direction is None) and (crossed_target_while_rising or crossed_target_while_falling):
            return value_index
    return None


def cast_to_string(
    elems: list,
    wrap_in_quotes: bool = False,
    conjunction: str = "",
) -> str:
    elems = flatten_list(list(elems))
    if len(elems) == 0:
        return ""
    elems = [f"`{elem}`" if wrap_in_quotes else str(elem) for elem in elems]
    conjunction = conjunction.strip()
    if conjunction == "":
        return ", ".join(elems)
    if len(elems) == 2:
        return f"{elems[0]} {conjunction} {elems[1]}"
    return ", ".join(elems[:-1]) + f" {conjunction} {elems[-1]}"


def get_preview_string(
    elems: list,
    preview_length: int | None = None,
) -> str:
    elems_preview = cast_to_string(
        elems=elems[:preview_length],
        wrap_in_quotes=False,
        conjunction=None,
    )
    return elems_preview + ("..." if len(elems) > preview_length else "")


def get_intersect_of_lists(
    list_a: list,
    list_b: list,
    sort_values: bool = False,
) -> list:
    """Find the intersection of two lists (optionally sorted)."""
    type_manager.ensure_type(
        param=list_a,
        valid_types=(list, numpy.ndarray),
    )
    type_manager.ensure_type(
        param=list_b,
        valid_types=(list, numpy.ndarray),
    )
    if (len(list_a) == 0) or (len(list_b) == 0): return []
    set_intersect = set(list_a) & set(list_b)
    return sorted(set_intersect) if sort_values else list(set_intersect)


def get_union_of_lists(
    list_a: list,
    list_b: list,
    sort_values: bool = False,
) -> list:
    """Find the union of two lists (optionally sorted)."""
    type_manager.ensure_type(
        param=list_a,
        valid_types=(list, numpy.ndarray),
    )
    type_manager.ensure_type(
        param=list_b,
        valid_types=(list, numpy.ndarray),
    )
    if (len(list_a) == 0) or (len(list_b) == 0): return list(list_a) + list(list_b)
    set_union = set(list_a) | set(list_b)
    return sorted(set_union) if sort_values else list(set_union)


def flatten_list(
    elems: list,
) -> list:
    """Flatten a nested list into a single list."""
    type_manager.ensure_type(
        param=elems,
        valid_types=(list, numpy.ndarray),
    )
    flat_elems = []
    for elem in list(elems):
        if isinstance(elem, (list, numpy.ndarray)):
            flat_elems.extend(list(flatten_list(elem)))
        else:
            flat_elems.append(elem)
    return flat_elems


## } MODULE
