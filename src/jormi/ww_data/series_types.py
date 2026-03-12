## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import array_checks

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

    x_values: numpy.ndarray
    y_values: numpy.ndarray

    def __post_init__(
        self,
    ):
        self._validate_data_array(self.x_values)
        self._validate_data_array(self.y_values)
        array_checks.ensure_same_shape(
            array_a=self.x_values,
            array_b=self.y_values,
        )

    @staticmethod
    def _validate_data_array(
        array: numpy.ndarray,
        rel_tol: float = 1e-2,
        abs_tol: float = 1e-9,
    ):
        array_checks.ensure_nonempty(array)
        array_checks.ensure_finite(array)
        array_checks.ensure_1d(array)
        value_range = numpy.max(array) - numpy.min(array)
        ref_value = max(
            1.0,  # clamp: prevents near-zero scale from inflating the ratio
            numpy.median(numpy.abs(array)),  # robust estimate of typical data magnitude
        )
        if (value_range / ref_value < rel_tol) and (value_range < abs_tol):
            raise ValueError("Data values are (nearly) identical.")

    @staticmethod
    def _validate_sigma_array(
        sigma_array: numpy.ndarray,
        ref_array: numpy.ndarray,
    ):
        array_checks.ensure_nonempty(sigma_array)
        array_checks.ensure_finite(sigma_array)
        array_checks.ensure_1d(sigma_array)
        array_checks.ensure_same_shape(
            array_a=sigma_array,
            array_b=ref_array,
        )
        if numpy.any(sigma_array <= 0):
            raise ValueError("Uncertainty array must be strictly positive.")

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

    x_sigmas: numpy.ndarray | None = None
    y_sigmas: numpy.ndarray | None = None

    def __post_init__(
        self,
    ):
        super().__post_init__()
        if self.x_sigmas is not None:
            self._validate_sigma_array(
                sigma_array=self.x_sigmas,
                ref_array=self.x_values,
            )
        if self.y_sigmas is not None:
            self._validate_sigma_array(
                sigma_array=self.y_sigmas,
                ref_array=self.y_values,
            )

    def y_weights(
        self,
    ) -> numpy.ndarray:
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

    x_p16_values: numpy.ndarray | None = None
    x_p84_values: numpy.ndarray | None = None
    y_p16_values: numpy.ndarray | None = None
    y_p84_values: numpy.ndarray | None = None

    def __post_init__(
        self,
    ):
        super().__post_init__()
        if (self.x_p16_values is None) != (self.x_p84_values is None):
            raise ValueError("`x_p16_values` and `x_p84_values` must be provided together or not at all.")
        if (self.y_p16_values is None) != (self.y_p84_values is None):
            raise ValueError("`y_p16_values` and `y_p84_values` must be provided together or not at all.")
        for sigma_array, ref_array in [
            (self.x_p16_values, self.x_values),
            (self.x_p84_values, self.x_values),
            (self.y_p16_values, self.y_values),
            (self.y_p84_values, self.y_values),
        ]:
            if sigma_array is not None:
                self._validate_sigma_array(
                    sigma_array=sigma_array,
                    ref_array=ref_array,
                )


## } MODULE
