## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from functools import cached_property

## third-party
import numpy
from typing import Any
from numpy.typing import NDArray

## local
from jormi.ww_validation import validate_arrays

##
## === DATA SERIES
##


@dataclass(frozen=True)
class DataSeries:
    """
    A 1D (x, y) data series with no uncertainty model.

    Fields
    ---
    - `x_values`:
        1D array of x-values; must be finite and non-constant.

    - `y_values`:
        1D array of y-values; must be finite, non-constant, and same length as `x_values`.
    """

    x_values: NDArray[Any]
    y_values: NDArray[Any]

    def __post_init__(
        self,
    ):
        ## validate each coordinate series independently
        self._ensure_data_array(array=self.x_values)
        self._ensure_data_array(array=self.y_values)
        ## validate paired x/y sampling
        validate_arrays.ensure_same_shape(
            array_a=self.x_values,
            array_b=self.y_values,
        )

    @staticmethod
    def _ensure_data_array(
        *,
        array: NDArray[Any],
        rel_tol: float = 1e-2,
        abs_tol: float = 1e-9,
    ):
        validate_arrays.ensure_nonempty(array)
        validate_arrays.ensure_finite(array)
        validate_arrays.ensure_1d(array)
        value_range = numpy.max(array) - numpy.min(array)
        ref_value = max(
            1.0,  # clamp: prevents near-zero scale from inflating the ratio
            numpy.median(
                numpy.abs(
                    array,
                ),
            ),  # robust estimate of typical data magnitude
        )
        if (value_range / ref_value < rel_tol) and (value_range < abs_tol):
            raise ValueError("data values are (nearly) identical.")

    @staticmethod
    def _ensure_sigma_array(
        *,
        sigma_array: NDArray[Any],
        ref_array: NDArray[Any],
    ):
        validate_arrays.ensure_nonempty(sigma_array)
        validate_arrays.ensure_finite(sigma_array)
        validate_arrays.ensure_1d(sigma_array)
        validate_arrays.ensure_same_shape(
            array_a=sigma_array,
            array_b=ref_array,
        )
        if numpy.any(sigma_array <= 0):
            raise ValueError("uncertainty values must be strictly positive.")

    @cached_property
    def num_points(
        self,
    ) -> int:
        return len(self.x_values)

    @cached_property
    def x_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.x_values),
            numpy.max(self.x_values),
        )

    @cached_property
    def y_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.y_values),
            numpy.max(self.y_values),
        )


##
## === GAUSSIAN SERIES
##


@dataclass(frozen=True)
class GaussianSeries(DataSeries):
    """
    A `DataSeries` with optional symmetric per-point Gaussian uncertainties.

    Assumes a normal error model (single sigma per point), which maps directly
    to inverse-variance weighting in least-squares fitting.

    Fields
    ---
    - `x_sigmas`:
        Optional 1D array of x-uncertainties; must be strictly positive.

    - `y_sigmas`:
        Optional 1D array of y-uncertainties; must be strictly positive.
    """

    x_sigmas: NDArray[Any] | None = None
    y_sigmas: NDArray[Any] | None = None

    def __post_init__(
        self,
    ):
        super().__post_init__()
        ## validate the optional uncertainty arrays against their data axes
        if self.x_sigmas is not None:
            self._ensure_sigma_array(
                sigma_array=self.x_sigmas,
                ref_array=self.x_values,
            )
        if self.y_sigmas is not None:
            self._ensure_sigma_array(
                sigma_array=self.y_sigmas,
                ref_array=self.y_values,
            )

    def y_weights(
        self,
    ) -> NDArray[Any]:
        """Inverse-variance weights (1/sigma^2) for weighted least-squares. Returns ones if no `y_sigmas`."""
        if self.y_sigmas is None:
            return numpy.ones_like(self.y_values, dtype=float)
        return 1.0 / numpy.square(self.y_sigmas)


##
## === DISTRIBUTION SERIES
##


@dataclass(frozen=True)
class DistributionSeries(DataSeries):
    """
    A `DataSeries` with optional asymmetric per-point uncertainties (p16/p84).

    Suitable for data where the uncertainty is not symmetric about the central value,
    e.g. Poisson-distributed counts or percentile-derived error bars. The p16 and p84
    percentiles correspond to +/-1 sigma for a Gaussian distribution.

    Fields
    ---
    - `x_p16_values`:
        Optional 1D array of lower x-uncertainties (16th percentile offset); must be strictly positive.

    - `x_p84_values`:
        Optional 1D array of upper x-uncertainties (84th percentile offset); must be strictly positive.

    - `y_p16_values`:
        Optional 1D array of lower y-uncertainties (16th percentile offset); must be strictly positive.

    - `y_p84_values`:
        Optional 1D array of upper y-uncertainties (84th percentile offset); must be strictly positive.

    Notes
    ---
    - `x_p16_values` and `x_p84_values` must be provided together or not at all.
    - `y_p16_values` and `y_p84_values` must be provided together or not at all.
    """

    x_p16_values: NDArray[Any] | None = None
    x_p84_values: NDArray[Any] | None = None
    y_p16_values: NDArray[Any] | None = None
    y_p84_values: NDArray[Any] | None = None

    def __post_init__(
        self,
    ):
        super().__post_init__()
        ## validate that asymmetric bounds are supplied in pairs
        if (self.x_p16_values is None) != (self.x_p84_values is None):
            raise ValueError("`x_p16_values` and `x_p84_values` must be provided together or not at all.")
        if (self.y_p16_values is None) != (self.y_p84_values is None):
            raise ValueError("`y_p16_values` and `y_p84_values` must be provided together or not at all.")
        ## validate each provided uncertainty array against its reference axis
        for sigma_array, ref_array in [
            (self.x_p16_values, self.x_values),
            (self.x_p84_values, self.x_values),
            (self.y_p16_values, self.y_values),
            (self.y_p84_values, self.y_values),
        ]:
            if sigma_array is not None:
                self._ensure_sigma_array(
                    sigma_array=sigma_array,
                    ref_array=ref_array,
                )


## } MODULE
