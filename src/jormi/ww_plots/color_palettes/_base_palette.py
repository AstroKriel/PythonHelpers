## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass

## third-party
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import numpy

## side-effect import: registers cmasher colormaps with matplotlib
import cmasher  # noqa: F401  # pyright: ignore[reportUnusedImport]

## local
from jormi.ww_validation import validate_types

##
## === INTERNAL HELPERS
##


def ensure_palette_range(
    palette_range: tuple[float, float],
) -> None:
    validate_types.ensure_ordered_pair(
        param=palette_range,
        param_name="palette_range",
        allow_none=False,
    )
    validate_types.ensure_in_bounds(
        param=palette_range[0],
        param_name="palette_range[0]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )
    validate_types.ensure_in_bounds(
        param=palette_range[1],
        param_name="palette_range[1]",
        allow_none=False,
        min_value=0.0,
        max_value=1.0,
    )


def resolve_palette(
    palette_name: str,
) -> mpl_colors.Colormap:
    validate_types.ensure_nonempty_string(
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
    validate_types.ensure_in_bounds(
        param=palette_min,
        min_value=0.0,
        max_value=1.0,
        param_name="palette_range[0]",
    )
    validate_types.ensure_in_bounds(
        param=palette_max,
        min_value=0.0,
        max_value=1.0,
        param_name="palette_range[1]",
    )
    validate_types.ensure_ordered_pair(
        param=palette_range,
        param_name="palette_range",
    )
    if (palette_min == 0.0) and (palette_max == 1.0):
        return palette_cmap
    ## look-up table size should have 256 values; match 8-bit color depth (this exceeds perceptual resolution)
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


def _make_builtin_palette(
    name: str,
    colors: list[str],
) -> tuple[str, mpl_colors.Colormap]:
    """Create a named colormap and return (name, cmap) for use as a _BUILTIN_PALETTES entry."""
    cmap = mpl_colors.LinearSegmentedColormap.from_list(
        name=name,
        colors=colors,
        N=256,
    )
    return (name, cmap)


_BUILTIN_PALETTES: dict[str, mpl_colors.Colormap] = dict(
    [
        _make_builtin_palette(
            name="blue-white-red",
            colors=["#024f92", "#067bf1", "#d4d4d4", "#f65d25", "#A41409"],
        ),
        _make_builtin_palette(
            name="white-brown",
            colors=["#fdfdfd", "#f49325", "#010101"],
        ),
        _make_builtin_palette(
            name="purple-white-green",
            colors=["#68287d", "#d0a7c7", "#f2f0e0", "#d5e370", "#275b0e"],
        ),
    ],
)

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
        hash=False,  # colormaps are not hashable
        compare=False,  # equality should reflect construction args
        repr=False,  # colormap repr is large and uninformative
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
