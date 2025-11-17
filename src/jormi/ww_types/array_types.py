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
):
    type_manager.ensure_type(
        param=array,
        valid_types=numpy.ndarray,
    )


def ensure_nonempty(
    array: numpy.ndarray,
):
    ensure_array(array)
    if array.size == 0:
        raise ValueError("Array is empty.")


def ensure_finite(
    array: numpy.ndarray,
):
    ensure_array(array)
    if not numpy.isfinite(array).all():
        raise ValueError("Array contains NaN or Inf.")


def ensure_shape(
    array: numpy.ndarray,
    expected_shape: tuple[int, ...],
):
    ensure_array(array)
    if array.shape != expected_shape:
        raise ValueError(f"Array should have shape {expected_shape}, got {array.shape}.")


def ensure_same_shape(
    array_a: numpy.ndarray,
    array_b: numpy.ndarray,
) -> None:
    ensure_array(array_a)
    ensure_array(array_b)
    if array_a.shape != array_b.shape:
        raise ValueError(f"Shape mismatch: {array_a.shape} vs {array_b.shape}")


def ensure_dim(
    array: numpy.ndarray,
    dim: int,
):
    ensure_array(array)
    type_manager.ensure_finite_int(
        param=dim,
        param_name="dim",
    )
    if dim < 1:
        raise ValueError("Input `dim`-param must be a positive integer.")
    if array.ndim != dim:
        raise ValueError(f"Array is not {dim}D.")


def ensure_1d(
    array: numpy.ndarray,
):
    ensure_dim(
        array=array,
        dim=1,
    )


def as_1d(
    array_like: list | numpy.ndarray,
    check_finite: bool = True,
) -> numpy.ndarray:
    array = numpy.asarray(array_like, numpy.float64)
    ensure_nonempty(array)
    if check_finite:
        ensure_finite(array)
    ensure_1d(array)
    return array.ravel()


## } MODULE
