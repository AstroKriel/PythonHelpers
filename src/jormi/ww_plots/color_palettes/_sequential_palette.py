## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

from dataclasses import dataclass

## third-party
import matplotlib.colors as mpl_colors

## local
## import directly from the module file (not via the package __init__) to avoid a static import cycle
from jormi.ww_plots.color_palettes._base_palette import (
    ColorPalette,
    ensure_palette_range,
    resolve_palette,
    subset_palette,
)
from jormi.ww_validation import validate_python_types

##
## === SEQUENTIAL PALETTE
##


@dataclass(frozen=True, kw_only=True)
class SequentialPalette(ColorPalette):
    """
    A continuous, single-direction color palette.

    Use `from_name` or `from_colors` to construct.
    """

    value_range: tuple[float, float]  # data-space (vmin, vmax)
    palette_range: tuple[float, float] = (0.0, 1.0)  # portion of palette to use, in [0, 1]

    def __post_init__(
        self,
    ) -> None:
        validate_python_types.ensure_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        ensure_palette_range(self.palette_range)

    @classmethod
    def from_name(
        cls,
        *,
        value_range: tuple[float, float],
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "SequentialPalette":
        return cls(
            value_range=value_range,
            palette_range=palette_range,
            _base_cmap=resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        value_range: tuple[float, float],
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "SequentialPalette":
        ## look-up table size should be 256 to match 8-bit color depth (this exceeds perceptual resolution)
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom-sequential-cmap",
            colors=colors,
            N=256,
        )
        return cls(
            value_range=value_range,
            palette_range=palette_range,
            _base_cmap=cmap,
        )

    @property
    def mpl_norm(
        self,
    ) -> mpl_colors.Normalize:
        value_min, value_max = float(self.value_range[0]), float(self.value_range[1])
        return mpl_colors.Normalize(
            vmin=value_min,
            vmax=value_max,
        )

    @property
    def mpl_cmap(
        self,
    ) -> mpl_colors.Colormap:
        return subset_palette(
            palette_cmap=self._base_cmap,
            palette_range=self.palette_range,
            palette_label="subset-sequential-cmap",
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "SequentialPalette":
        return dataclasses.replace(self, value_range=value_range)


## } MODULE
