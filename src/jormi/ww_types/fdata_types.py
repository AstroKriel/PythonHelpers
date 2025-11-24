## { MODULE

##
## === DEPENDENCIES
##

import numpy

from dataclasses import dataclass
from typing import TypeAlias

from jormi.ww_types import type_manager, array_checks


##
## === TYPE ALIASES
##

FieldArrayData: TypeAlias = numpy.ndarray


##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldArray:
    """
    Generic field array with component and spatial dimension metadata.

    Conventions
    ----------
    - `num_ranks`:
        Number of component axes:
          * 0 -> scalar (no component axis)
          * 1 -> vector (one component axis)
          * 2 -> rank-2 tensor (two component axes)
          * ...
    - `num_comps`:
        Total number of components at each spatial location.
        A scalar field has `num_comps == 1` and a vector has `num_comps > 1`.
    - `num_sdims`:
        Number of spatial dimensions.

    The first `num_ranks` axes are component axes; the trailing `num_sdims`
    axes are spatial.
    """

    data: FieldArrayData
    num_ranks: int
    num_comps: int
    num_sdims: int
    param_name: str = "<field_array>"

    def __post_init__(
        self,
    ) -> None:
        type_manager.ensure_finite_int(
            param=self.num_comps,
            param_name=f"{self.param_name}.num_comps",
            allow_none=False,
            require_positive=True,
        )
        type_manager.ensure_finite_int(
            param=self.num_sdims,
            param_name=f"{self.param_name}.num_sdims",
            allow_none=False,
            require_positive=True,
        )
        type_manager.ensure_finite_int(
            param=self.num_ranks+1,
            param_name=f"{self.param_name}.num_ranks",
            allow_none=False,
            require_positive=True,
        )
        if (self.num_ranks == 0) and (self.num_comps != 1):
            raise ValueError(
                f"`{self.param_name}` has num_ranks=0 (scalar) but num_comps={self.num_comps};"
                f" expected num_comps == 1.",
            )
        array_checks.ensure_ndim(
            array=self.data,
            param_name=self.param_name,
            num_dims=self._total_num_of_dims(),
        )
        self._ensure_shape_matches_metadata()

    @property
    def shape(
        self,
    ) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def is_scalar(
        self,
    ) -> bool:
        return self.num_ranks == 0

    @property
    def is_vector(
        self,
    ) -> bool:
        return (self.num_ranks == 1) and (self.num_comps > 1)

    @property
    def is_tensor(
        self,
    ) -> bool:
        return self.num_ranks >= 2

    @property
    def dims_shape(
        self,
    ) -> tuple[int, ...]:
        """Return the spatial part of the shape."""
        if self.num_ranks == 0:
            return self.shape
        return self.shape[self.num_ranks :]

    @property
    def comps_shape(
        self,
    ) -> tuple[int, ...]:
        """Return the component part of the shape."""
        if self.num_ranks == 0:
            return ()
        return self.shape[: self.num_ranks]

    def _total_num_of_dims(
        self,
    ) -> int:
        return self.num_ranks + self.num_sdims

    def _ensure_shape_matches_metadata(
        self,
    ) -> None:
        if self.data.ndim != self._total_num_of_dims():
            raise ValueError(
                f"`{self.param_name}` must have ndim == num_ranks + num_sdims:"
                f" ({self.num_ranks} + {self.num_sdims} = {self._total_num_of_dims()}),"
                f" but got shape={self.data.shape}.",
            )
        if self.num_ranks == 0:
            ## already ensured ndim == num_sdims; nothing more to check
            return
        ## for num_ranks > 0, ensure product of component-axis sizes matches num_comps
        comps_shape = self.comps_shape
        num_comps_from_shape = 1
        for axis_size in comps_shape:
            num_comps_from_shape *= axis_size
        if num_comps_from_shape != self.num_comps:
            raise ValueError(
                f"`{self.param_name}` has num_comps={self.num_comps}, but component axes"
                f" {comps_shape} imply num_comps={num_comps_from_shape}.",
            )


##
## === TYPE VALIDATION HELPERS
##


def ensure_field_array(
    field_array: FieldArray,
    *,
    param_name: str = "<field_array>",
) -> None:
    """Ensure `field_array` is a `FieldArray` instance."""
    if not isinstance(field_array, FieldArray):
        raise TypeError(
            f"`{param_name}` must be a `FieldArray` instance; got type={type(field_array)}.",
        )


def ensure_field_array_metadata(
    field_array: FieldArray,
    *,
    num_comps: int | None = None,
    num_sdims: int | None = None,
    num_ranks: int | None = None,
    param_name: str = "<field_array>",
) -> None:
    """
    Ensure `field_array` matches the requested metadata.

    Any of `num_comps`, `num_sdims`, or `num_ranks` can be left as `None`
    to skip that check.
    """
    ensure_field_array(
        field_array=field_array,
        param_name=param_name,
    )
    if (num_comps is not None) and (field_array.num_comps != num_comps):
        raise ValueError(
            f"`{param_name}` must have num_comps={num_comps},"
            f" but got num_comps={field_array.num_comps}.",
        )
    if (num_sdims is not None) and (field_array.num_sdims != num_sdims):
        raise ValueError(
            f"`{param_name}` must have num_sdims={num_sdims},"
            f" but got num_sdims={field_array.num_sdims}.",
        )
    if (num_ranks is not None) and (field_array.num_ranks != num_ranks):
        raise ValueError(
            f"`{param_name}` must have num_ranks={num_ranks},"
            f" but got num_ranks={field_array.num_ranks}.",
        )


def ensure_sarray_3d(
    field_array: FieldArray,
    *,
    param_name: str = "<field_array>",
) -> None:
    """Ensure `field_array` is a 3D scalar field array."""
    ensure_field_array_metadata(
        field_array=field_array,
        num_comps=1,
        num_sdims=3,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_varray_3d(
    field_array: FieldArray,
    *,
    param_name: str = "<field_array>",
) -> None:
    """Ensure `field_array` is a 3D vector field array with 3 components."""
    ensure_field_array_metadata(
        field_array=field_array,
        num_comps=3,
        num_sdims=3,
        num_ranks=1,
        param_name=param_name,
    )


def ensure_r2tarray_3d(
    field_array: FieldArray,
    *,
    param_name: str = "<field_array>",
) -> None:
    """Ensure `field_array` is a 3D rank-2 tensor field array with 3x3 components."""
    ensure_field_array_metadata(
        field_array=field_array,
        num_comps=9,
        num_sdims=3,
        num_ranks=2,
        param_name=param_name,
    )


## } MODULE
