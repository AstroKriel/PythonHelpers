## { MODULE

##
## === DEPENDENCIES
##

import numpy

from functools import cached_property
from dataclasses import dataclass

from jormi.ww_types import array_checks

##
## === DATA TYPES
##


@dataclass(frozen=True)
class DataSeries:
    x_data_array: numpy.ndarray
    y_data_array: numpy.ndarray
    x_sigma_array: numpy.ndarray | None = None
    y_sigma_array: numpy.ndarray | None = None

    def __post_init__(
        self,
    ):
        self._validate_data_array(self.x_data_array)
        self._validate_data_array(self.y_data_array)
        array_checks.ensure_same_shape(
            array_a=self.x_data_array,
            array_b=self.y_data_array,
        )
        if self.x_sigma_array is not None:
            self._validate_sigma_array(
                sigma_array=self.x_sigma_array,
                ref_array=self.x_data_array,
            )
        if self.y_sigma_array is not None:
            self._validate_sigma_array(
                sigma_array=self.y_sigma_array,
                ref_array=self.y_data_array,
            )

    @staticmethod
    def _validate_data_array(
        array: numpy.ndarray,
    ):
        array_checks.ensure_nonempty(array)
        array_checks.ensure_finite(array)
        array_checks.ensure_1d(array)
        value_range = numpy.max(array) - numpy.min(array)
        ref_value = max(
            1.0,
            numpy.median(numpy.abs(array)),
        )
        if (value_range / ref_value < 1e-2) and (value_range < 1e-9):
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
        return len(self.x_data_array)

    @cached_property
    def x_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.x_data_array),
            numpy.max(self.x_data_array),
        )

    @cached_property
    def y_bounds(
        self,
    ) -> tuple[float, float]:
        return (
            numpy.min(self.y_data_array),
            numpy.max(self.y_data_array),
        )

    @cached_property
    def has_x_uncertainty(
        self,
    ) -> bool:
        return self.x_sigma_array is not None

    @cached_property
    def has_y_uncertainty(
        self,
    ) -> bool:
        return self.y_sigma_array is not None

    def y_weights(
        self,
    ) -> numpy.ndarray:
        if self.y_sigma_array is None:
            return numpy.ones_like(self.y_data_array, dtype=float)
        return 1.0 / numpy.square(self.y_sigma_array)
    

## } MODULE
