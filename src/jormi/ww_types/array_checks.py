## { MODULE

##
## === DEPENDENCIES
##

import numpy

from jormi.ww_types import type_manager

##
## === FUNCTIONS
##


def ensure_array(
    array: numpy.ndarray,
    *,
    param_name: str = "<array>",
) -> None:
    """Ensure `array` is a NumPy ndarray."""
    type_manager.ensure_ndarray(
        param=array,
        param_name=param_name,
    )


def ensure_nonempty(
    array: numpy.ndarray,
    *,
    param_name: str = "<array>",
) -> None:
    """Ensure `array` is a non-empty NumPy ndarray."""
    ensure_array(
        array=array,
        param_name=param_name,
    )
    if array.size == 0:
        raise ValueError(f"`{param_name}` is empty.")


def ensure_finite(
    array: numpy.ndarray,
    *,
    param_name: str = "<array>",
) -> None:
    """Ensure `array` is a finite NumPy ndarray (no NaN/Inf)."""
    ensure_array(
        array=array,
        param_name=param_name,
    )
    if not numpy.isfinite(array).all():
        raise ValueError(f"`{param_name}` contains NaN or Inf.")


def ensure_shape(
    array: numpy.ndarray,
    *,
    expected_shape: tuple[int, ...],
    param_name: str = "<array>",
) -> None:
    """Ensure `array` has the given `expected_shape`."""
    ensure_array(
        array=array,
        param_name=param_name,
    )
    type_manager.ensure_tuple_of_ints(
        param=expected_shape,
        param_name="expected_shape",
    )
    if array.shape != expected_shape:
        raise ValueError(
            f"`{param_name}` should have shape {expected_shape}, got {array.shape}.",
        )


def ensure_same_shape(
    *,
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
    param_name_a: str = "<array_a>",
    param_name_b: str = "<array_b>",
) -> None:
    """Ensure `array_a` and `array_b` are ndarrays with identical shape."""
    ensure_array(
        array=array_a,
        param_name=param_name_a,
    )
    ensure_array(
        array=array_b,
        param_name=param_name_b,
    )
    if array_a.shape != array_b.shape:
        raise ValueError(
            f"Shape mismatch: `{param_name_a}` {array_a.shape} vs "
            f"`{param_name_b}` {array_b.shape}.",
        )


def ensure_dims(
    array: numpy.ndarray,
    num_dims: int,
    *,
    param_name: str = "<array>",
) -> None:
    """Ensure `array` is a NumPy ndarray with array.ndim == num_dims."""
    ensure_array(
        array=array,
        param_name=param_name,
    )
    type_manager.ensure_finite_int(
        param=num_dims,
        param_name="num_dims",
        require_positive=True,
    )
    if array.ndim != num_dims:
        raise ValueError(
            f"`{param_name}` must be {num_dims}-dimensional; got ndim={array.ndim}.",
        )


def ensure_1d(
    array: numpy.ndarray,
    *,
    param_name: str = "<array>",
) -> None:
    """Ensure `array` is a 1D NumPy ndarray."""
    ensure_dims(
        array=array,
        num_dims=1,
        param_name=param_name,
    )


def as_1d(
    array_like: tuple | list | numpy.ndarray,
    *,
    param_name: str = "<array_like>",
    check_finite: bool = True,
) -> numpy.ndarray:
    """Convert `array_like` to a 1D ndarray[float64]."""
    type_manager.ensure_not_none(
        param=array_like,
        param_name=param_name,
    )
    array = numpy.asarray(array_like, numpy.float64)
    ensure_nonempty(
        array=array,
        param_name=param_name,
    )
    if check_finite:
        ensure_finite(
            array=array,
            param_name=param_name,
        )
    ensure_1d(
        array=array,
        param_name=param_name,
    )
    return array.ravel()


## } MODULE
