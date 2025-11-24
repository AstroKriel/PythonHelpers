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
        ## NOTE: `num_ranks+1` is used to enforce num_ranks >= 0
        type_manager.ensure_finite_int(
            param=self.num_ranks + 1,
            param_name=f"{self.param_name}.num_ranks",
            allow_none=False,
            require_positive=True,
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


@dataclass(frozen=True, init=False)
class ScalarFieldData(FieldData):
    """3D scalar field data: shape (Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: FieldArray,
        param_name: str = "<sdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=0,
            num_comps=1,
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class VectorFieldData(FieldData):
    """3D vector field data: shape (3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: FieldArray,
        param_name: str = "<vdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=1,
            num_comps=3,
            num_sdims=3,
            param_name=param_name,
        )


@dataclass(frozen=True, init=False)
class Rank2TensorData(FieldData):
    """3D rank-2 tensor field data: shape (3, 3, Nx, Ny, Nz)."""

    def __init__(
        self,
        *,
        farray: FieldArray,
        param_name: str = "<r2tdata>",
    ) -> None:
        super().__init__(
            farray=farray,
            num_ranks=2,
            num_comps=9,
            num_sdims=3,
            param_name=param_name,
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


def ensure_3d_sdata(
    sdata: FieldData,
    *,
    param_name: str = "<sdata>",
) -> None:
    """Ensure `sdata` is *specifically* ScalarFieldData with 3D scalar layout."""
    if not isinstance(sdata, ScalarFieldData):
        raise TypeError(
            f"`{param_name}` must be ScalarFieldData; got type={type(sdata)}.",
        )
    ensure_fdata_metadata(
        fdata=sdata,
        num_comps=1,
        num_sdims=3,
        num_ranks=0,
        param_name=param_name,
    )


def ensure_3d_vdata(
    vdata: FieldData,
    *,
    param_name: str = "<vdata>",
) -> None:
    """Ensure `vdata` is *specifically* VectorFieldData with 3 components in 3D."""
    if not isinstance(vdata, VectorFieldData):
        raise TypeError(
            f"`{param_name}` must be VectorFieldData; got type={type(vdata)}.",
        )
    ensure_fdata_metadata(
        fdata=vdata,
        num_comps=3,
        num_sdims=3,
        num_ranks=1,
        param_name=param_name,
    )


def ensure_3d_r2tdata(
    r2tdata: FieldData,
    *,
    param_name: str = "<r2tdata>",
) -> None:
    """Ensure `r2tdata` is *specifically* Rank2TensorData with 3x3 components in 3D."""
    if not isinstance(r2tdata, Rank2TensorData):
        raise TypeError(
            f"`{param_name}` must be Rank2TensorData; got type={type(r2tdata)}.",
        )
    ensure_fdata_metadata(
        fdata=r2tdata,
        num_comps=9,
        num_sdims=3,
        num_ranks=2,
        param_name=param_name,
    )


##
## === NDARRAY VALIDATION
##


def ensure_3d_sarray(
    sarray: numpy.ndarray,
    *,
    param_name: str = "<sarray>",
) -> None:
    """Ensure `sarray` is a 3D scalar ndarray with shape (Nx, Ny, Nz)."""
    array_checks.ensure_dims(
        array=sarray,
        param_name=param_name,
        num_dims=3,
    )


def ensure_3d_varray(
    varray: numpy.ndarray,
    *,
    param_name: str = "<varray>",
) -> None:
    """Ensure `varray` is a 4D vector ndarray with leading axis of length 3."""
    array_checks.ensure_dims(
        array=varray,
        param_name=param_name,
        num_dims=4,
    )
    if varray.shape[0] != 3:
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={varray.shape}.",
        )


def ensure_3d_r2tarray(
    r2tarray: numpy.ndarray,
    *,
    param_name: str = "<r2tarray>",
) -> None:
    """Ensure `r2tarray` is a 5D rank-2 tensor ndarray with two leading axes of length 3."""
    array_checks.ensure_dims(
        array=r2tarray,
        param_name=param_name,
        num_dims=5,
    )
    if (r2tarray.shape[0] != 3) or (r2tarray.shape[1] != 3):
        raise ValueError(
            f"`{param_name}` must have shape"
            f" (3, 3, num_cells_x, num_cells_y, num_cells_z);"
            f" got shape={r2tarray.shape}.",
        )


##
## === NDARRAY NORMALISERS
##


def as_3d_sarray(
    sdata: ScalarFieldData | numpy.ndarray,
    *,
    param_name: str = "<sdata>",
) -> numpy.ndarray:
    """
    Normalise `sdata` to a 3D scalar ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (Nx, Ny, Nz), or
      - ScalarFieldData (3D scalar field data).
    """
    if isinstance(sdata, ScalarFieldData):
        farray = sdata.farray
        ensure_3d_sarray(
            sarray=farray,
            param_name="<sdata.farray>",
        )
        return farray
    ensure_3d_sarray(
        sarray=sdata,
        param_name=param_name,
    )
    return sdata


def as_3d_varray(
    vdata: VectorFieldData | numpy.ndarray,
    *,
    param_name: str = "<vdata>",
) -> numpy.ndarray:
    """
    Normalise `vdata` to a 3D vector ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (3, Nx, Ny, Nz), or
      - VectorFieldData (3D vector field data).
    """
    if isinstance(vdata, VectorFieldData):
        farray = vdata.farray
        ensure_3d_varray(
            varray=farray,
            param_name="<vdata.farray>",
        )
        return farray
    ensure_3d_varray(
        varray=vdata,
        param_name=param_name,
    )
    return vdata


def as_3d_r2tarray(
    r2tdata: Rank2TensorData | numpy.ndarray,
    *,
    param_name: str = "<r2tdata>",
) -> numpy.ndarray:
    """
    Normalise `r2tdata` to a 3D rank-2 tensor ndarray and validate its structure.

    Accepts either:
      - raw ndarray of shape (3, 3, Nx, Ny, Nz), or
      - Rank2TensorData (3D rank-2 tensor field data).
    """
    if isinstance(r2tdata, Rank2TensorData):
        farray = r2tdata.farray
        ensure_3d_r2tarray(
            r2tarray=farray,
            param_name="<r2tdata.farray>",
        )
        return farray
    ensure_3d_r2tarray(
        r2tarray=r2tdata,
        param_name=param_name,
    )
    return r2tdata


## } MODULE
