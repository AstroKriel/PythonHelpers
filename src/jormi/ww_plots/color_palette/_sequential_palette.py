## { MODULE

##
## === DEPENDENCIES
##

import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_plots.color_palette import _base_palette

##
## === CLASS
##


@dataclass(frozen=True, kw_only=True)
class SequentialPalette(_base_palette.ColorPalette):
    """
    A continuous, single-direction color palette.

    Fields
    ---
    - `value_range`:
        Data-space (vmin, vmax) tuple.
    - `palette_range`:
        Portion of the palette to use, as a (min, max) tuple in [0, 1].
    - `_base_colormap`:
        Internal: the pre-built base colormap. Use from_name or from_colors.
    """
    value_range: tuple[float, float]
    palette_range: tuple[float, float] = (0.0, 1.0)
    _base_colormap: mpl_colors.Colormap = dataclasses.field(
        hash=False,
        compare=False,
        repr=False,
    )

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
            _base_colormap=_base_palette.resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        value_range: tuple[float, float],
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "SequentialPalette":
        base = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom",
            colors=colors,
            N=256,
        )
        return cls(
            value_range=value_range,
            palette_range=palette_range,
            _base_colormap=base,
        )

    @property
    def _mpl_norm(self) -> mpl_colors.Normalize:
        vmin, vmax = float(self.value_range[0]), float(self.value_range[1])
        return mpl_colors.Normalize(
            vmin=vmin,
            vmax=vmax,
        )

    @property
    def _mpl_colormap(self) -> mpl_colors.Colormap:
        return _base_palette.subset_palette(
            palette=self._base_colormap,
            palette_range=self.palette_range,
            name="sequential",
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "SequentialPalette":
        return dataclasses.replace(self, value_range=value_range)


## } MODULE
