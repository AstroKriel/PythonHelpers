## { MODULE

##
## === DEPENDENCIES
##

import numpy
import dataclasses
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

## side-effect import: registers cmasher colormaps with matplotlib
import cmasher  # noqa: F401

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jormi.ww_types import type_checks

##
## === INTERNAL HELPERS
##


def validate_palette_range(
    palette_range: tuple[float, float],
) -> None:
    type_checks.ensure_ordered_pair(
        param=palette_range,
        param_name="palette_range",
        allow_none=False,
    )
    type_checks.ensure_in_bounds(
        param=palette_range[0],
        param_name="palette_range[0]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    type_checks.ensure_in_bounds(
        param=palette_range[1],
        param_name="palette_range[1]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )


def resolve_palette(
    palette_name: str,
) -> mpl_colors.Colormap:
    type_checks.ensure_nonempty_string(
        param=palette_name,
        param_name="palette_name",
    )
    if palette_name in _BUILTIN_PALETTES:
        return _BUILTIN_PALETTES[palette_name]
    return mpl_cm.get_cmap(palette_name)


def subset_palette(
    *,
    palette_cmap: mpl_colors.Colormap,
    palette_range: tuple[float, float],
    palette_label: str,
) -> mpl_colors.Colormap:
    palette_min, palette_max = palette_range
    type_checks.ensure_in_bounds(
        param=palette_min,
        min_value=0.0,
        max_value=1.0,
        param_name="palette_range[0]",
    )
    type_checks.ensure_in_bounds(
        param=palette_max,
        min_value=0.0,
        max_value=1.0,
        param_name="palette_range[1]",
    )
    type_checks.ensure_ordered_pair(
        param=palette_range,
        param_name="palette_range",
    )
    if (palette_min == 0.0) and (palette_max == 1.0):
        return palette_cmap
    ## look-up table size should be 256 to match 8-bit color depth (this exceeds perceptual resolution)
    sampled_colors = palette_cmap(
        numpy.linspace(
            start=palette_min,
            stop=palette_max,
            num=256,
        ),
    )
    return mpl_colors.LinearSegmentedColormap.from_list(
        name=palette_label,
        colors=sampled_colors,
        N=256,
    )


##
## === BUILTIN PALETTES
##

_BUILTIN_PALETTES: dict[str, mpl_colors.Colormap] = {
    "blue-white-red":
    mpl_colors.LinearSegmentedColormap.from_list(
        name="blue-white-red",
        colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        N=256,
    ),
    "white-brown":
    mpl_colors.LinearSegmentedColormap.from_list(
        name="white-brown",
        colors=["#fdfdfd", "#f49325", "#010101"],
        N=256,
    ),
    "purple-white-green":
    mpl_colors.LinearSegmentedColormap.from_list(
        name="purple-white-green",
        colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        N=256,
    ),
}

##
## === BASE PALETTE
##


@dataclass(frozen=True, kw_only=True)
class ColorPalette(ABC):
    """
    Abstract base for all color palette types.

    Subclasses must implement `mpl_norm` and `mpl_cmap`, and expose
    a `palette_range` field.
    """

    _base_cmap: mpl_colors.Colormap = dataclasses.field(
        hash=False,  # colormaps are not hashable; including this field would raise TypeError in __hash__
        compare=False,  # equality should reflect construction args, not colormap object identity
        repr=False,  # colormap repr is large and uninformative; exclude to keep __repr__ clean
    )

    @property
    @abstractmethod
    def mpl_norm(
        self,
    ) -> mpl_colors.Normalize:
        ...

    @property
    @abstractmethod
    def mpl_cmap(
        self,
    ) -> mpl_colors.Colormap:
        ...

    def with_palette_range(
        self,
        palette_range: tuple[float, float],
    ) -> "ColorPalette":
        return dataclasses.replace(self, palette_range=palette_range)


## } MODULE
