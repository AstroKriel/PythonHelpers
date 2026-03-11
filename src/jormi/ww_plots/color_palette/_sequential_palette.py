## { MODULE

##
## === DEPENDENCIES
##

import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_types import type_checks
from jormi.ww_plots.color_palette import _base_palette

##
## === SEQUENTIAL PALETTE
##


@dataclass(frozen=True, kw_only=True)
class SequentialPalette(_base_palette.ColorPalette):
    """
    A continuous, single-direction color palette.

    Use `from_name` or `from_colors` to construct.
    """

    value_range: tuple[float, float]  # data-space (vmin, vmax)
    palette_range: tuple[float, float] = (0.0, 1.0)  # portion of palette to use, in [0, 1]

    def __post_init__(
        self,
    ) -> None:
        type_checks.ensure_ordered_pair(
            param=self.value_range,
            param_name="value_range",
        )
        _base_palette.validate_palette_range(self.palette_range)

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
            _cmap=_base_palette.resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        value_range: tuple[float, float],
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "SequentialPalette":
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom",
            colors=colors,
            N=256,  # LUT size: 256 matches 8-bit color depth and exceeds perceptual resolution
        )
        return cls(
            value_range=value_range,
            palette_range=palette_range,
            _cmap=cmap,
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
        return _base_palette.subset_palette(
            palette_cmap=self._cmap,
            palette_range=self.palette_range,
            palette_label="subset-sequential-cmap",
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "SequentialPalette":
        return dataclasses.replace(self, value_range=value_range)


## } MODULE
