## { MODULE

##
## === DEPENDENCIES
##

import numpy
import cmasher
import dataclasses
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

from abc import ABC, abstractmethod

from jormi.ww_types import type_checks

##
## === INTERNAL HELPERS
##


def _ensure_in_unit_interval(
    value: float,
    *,
    param_name: str,
) -> None:
    type_checks.ensure_finite_float(
        param=value,
        param_name=param_name,
        allow_none=False,
    )
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"`{param_name}` must lie in [0, 1], got {value}.")


def _ensure_ordered_pair(
    value_pair: tuple[float | int, float | int],
    *,
    param_name: str,
) -> None:
    min_value, max_value = value_pair
    type_checks.ensure_finite_scalar(
        param=min_value,
        param_name=f"{param_name}[0]",
        allow_none=False,
    )
    type_checks.ensure_finite_scalar(
        param=max_value,
        param_name=f"{param_name}[1]",
        allow_none=False,
    )
    if not (float(min_value) <= float(max_value)):
        raise ValueError(f"`{param_name}` must satisfy [0] <= [1], got {value_pair}.")


def _get_base_palette(
    palette_name: str,
) -> mpl_colors.Colormap:
    type_checks.ensure_nonempty_string(
        param=palette_name,
        param_name="palette_name",
    )
    if palette_name in _BUILTIN_PALETTES:
        return _BUILTIN_PALETTES[palette_name]
    try:
        return mpl_cm.get_cmap(palette_name)
    except ValueError:
        return cmasher.get_cmap(palette_name)


def _subset_palette(
    palette: mpl_colors.Colormap,
    *,
    palette_range: tuple[float, float],
    name: str,
) -> mpl_colors.Colormap:
    p_min, p_max = palette_range
    _ensure_in_unit_interval(value=p_min, param_name="palette_range[0]")
    _ensure_in_unit_interval(value=p_max, param_name="palette_range[1]")
    _ensure_ordered_pair(palette_range, param_name="palette_range")
    if (p_min == 0.0) and (p_max == 1.0):
        return palette
    sample = palette(numpy.linspace(p_min, p_max, 256))
    return mpl_colors.LinearSegmentedColormap.from_list(
        name=f"{name}_sub",
        colors=sample,
        N=256,
    )


def _create_norm(
    *,
    value_range: tuple[float, float],
    mid_value: float | None,
) -> mpl_colors.Normalize:
    _ensure_ordered_pair(value_range, param_name="value_range")
    vmin, vmax = float(value_range[0]), float(value_range[1])
    if mid_value is None:
        return mpl_colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )
    type_checks.ensure_finite_float(
        param=mid_value,
        param_name="mid_value",
        allow_none=False,
    )
    vcenter = float(mid_value)
    if not (vmin < vcenter < vmax):
        raise ValueError(f"`mid_value` must satisfy vmin < mid_value < vmax, got ({vmin}, {vcenter}, {vmax}).")
    return mpl_colors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=vcenter,
        vmax=vmax,
    )


##
## === BUILTIN PALETTES
##


_BUILTIN_PALETTES: dict[str, mpl_colors.Colormap] = {
    "blue-red": mpl_colors.LinearSegmentedColormap.from_list(
        name="blue-red",
        colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        N=256,
    ),
    "white-brown": mpl_colors.LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#fdfdfd", "#f49325", "#010101"],
        N=256,
    ),
    "purple-green": mpl_colors.LinearSegmentedColormap.from_list(
        name="purple-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    ),
}

##
## === BASE CLASS
##


class ColourPalette(ABC):
    """
    Abstract base for all colour palette types.

    Subclasses must implement `_mpl_norm`. All subclasses are expected to
    expose the fields `palette_name`, `palette_range`, and `colours`, and
    a `value_range` property or field.
    """

    @property
    @abstractmethod
    def _mpl_norm(self) -> mpl_colors.Normalize: ...

    @property
    def _mpl_colormap(self) -> mpl_colors.Colormap:
        if self.colours is not None:  # type: ignore[attr-defined]
            base = mpl_colors.LinearSegmentedColormap.from_list(
                name=self.palette_name,  # type: ignore[attr-defined]
                colors=list(self.colours),  # type: ignore[attr-defined]
                N=256,
            )
        else:
            base = _get_base_palette(self.palette_name)  # type: ignore[attr-defined]
        return _subset_palette(
            palette=base,
            palette_range=self.palette_range,  # type: ignore[attr-defined]
            name=self.palette_name,  # type: ignore[attr-defined]
        )

    def with_palette_range(
        self,
        palette_range: tuple[float, float],
    ) -> "ColourPalette":
        return dataclasses.replace(self, palette_range=palette_range)  # type: ignore[call-overload]


## } MODULE
