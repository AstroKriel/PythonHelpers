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

FieldArray: TypeAlias = numpy.ndarray

##
## === DATA CLASSES
##


@dataclass(frozen=True)
class FieldData:
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

    farray: FieldArray
    num_ranks: int
    num_comps: int
    num_sdims: int
    param_name: str = "<fdata>"

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
            param=self.num_ranks,
            param_name=f"{self.param_name}.num_ranks",
            allow_none=False,
            require_positive=True,
            allow_zero=True,
        )
        if (self.num_ranks == 0) and (self.num_comps != 1):
            raise ValueError(
                f"`{self.param_name}` has num_ranks=0 (scalar) but num_comps={self.num_comps};"
                f" expected num_comps == 1.",
            )
        array_checks.ensure_dims(
            array=self.farray,
            param_name=self.param_name,
            num_dims=self._total_num_dims(),
        )
        self._ensure_shape_matches_metadata()

    @property
    def shape(
        self,
    ) -> tuple[int, ...]:
        return tuple(self.farray.shape)

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
    def sdims_shape(
        self,
    ) -> tuple[int, ...]:
        """Return the spatial part of the shape."""
        if self.num_ranks == 0:
            return self.shape
        return self.shape[self.num_ranks:]

    @property
    def comps_shape(
        self,
    ) -> tuple[int, ...]:
        """Return the component part of the shape."""
        if self.num_ranks == 0:
            return ()
        return self.shape[:self.num_ranks]

    def _total_num_dims(
        self,
    ) -> int:
        return self.num_ranks + self.num_sdims

    def _ensure_shape_matches_metadata(
        self,
    ) -> None:
        if self.farray.ndim != self._total_num_dims():
            raise ValueError(
                f"`{self.param_name}` must have ndim == num_ranks + num_sdims:"
                f" ({self.num_ranks} + {self.num_sdims} = {self._total_num_dims()}),"
                f" but got shape={self.farray.shape}.",
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
## === TYPE VALIDATION
##


def ensure_fdata(
    fdata: FieldData,
    *,
    param_name: str = "<fdata>",
) -> None:
    """Ensure `fdata` is a `FieldData` instance."""
    if not isinstance(fdata, FieldData):
        raise TypeError(
            f"`{param_name}` must be a `FieldData` instance; got type={type(fdata)}.",
        )


def ensure_fdata_metadata(
    fdata: FieldData,
    *,
    num_comps: int | None = None,
    num_sdims: int | None = None,
    num_ranks: int | None = None,
    param_name: str = "<fdata>",
) -> None:
    """
    Ensure `fdata` matches the requested metadata.

    Any of `num_comps`, `num_sdims`, or `num_ranks` can be left as `None`
    to skip that check.
    """
    ensure_fdata(
        fdata=fdata,
        param_name=param_name,
    )
    if (num_comps is not None) and (fdata.num_comps != num_comps):
        raise ValueError(
            f"`{param_name}` must have num_comps={num_comps},"
            f" but got num_comps={fdata.num_comps}.",
        )
    if (num_sdims is not None) and (fdata.num_sdims != num_sdims):
        raise ValueError(
            f"`{param_name}` must have num_sdims={num_sdims},"
            f" but got num_sdims={fdata.num_sdims}.",
        )
    if (num_ranks is not None) and (fdata.num_ranks != num_ranks):
        raise ValueError(
            f"`{param_name}` must have num_ranks={num_ranks},"
            f" but got num_ranks={fdata.num_ranks}.",
        )


## } MODULE
