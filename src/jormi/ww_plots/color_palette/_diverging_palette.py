## { MODULE

##
## === DEPENDENCIES
##

import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_plots.colour_palette._base import ColourPalette, _create_norm

##
## === CLASS
##


@dataclass(frozen=True, kw_only=True)
class DivergingPalette(ColourPalette):
    """
    A continuous, two-sided colour palette anchored at a midpoint.

    Parameters
    ----------
    value_range:
        Data-space (vmin, vmax) tuple.
    mid_value:
        Data-space midpoint, which maps to the centre of the palette.
        Must satisfy vmin < mid_value < vmax.
    palette_name:
        Name of the colour palette. Can be a Matplotlib name, a cmasher name,
        or one of the jormi built-in names ("blue-red", "white-brown", "purple-green").
    palette_range:
        Portion of the palette to use, as a (min, max) tuple in [0, 1].
    colours:
        Optional list of colours to build a custom palette from. If provided,
        palette_name is used only as a label.
    """
    value_range: tuple[float, float]
    mid_value: float
    palette_name: str = "blue-red"
    palette_range: tuple[float, float] = (0.0, 1.0)
    colours: tuple[str, ...] | None = None

    @classmethod
    def from_colours(
        cls,
        *,
        value_range: tuple[float, float],
        mid_value: float,
        colours: list[str],
        palette_name: str = "custom",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DivergingPalette":
        return cls(
            value_range=value_range,
            mid_value=mid_value,
            colours=tuple(colours),
            palette_name=palette_name,
            palette_range=palette_range,
        )

    @property
    def _mpl_norm(self) -> mpl_colors.Normalize:
        return _create_norm(
            value_range=self.value_range,
            mid_value=self.mid_value,
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "DivergingPalette":
        return dataclasses.replace(self, value_range=value_range)

    def with_mid_value(
        self,
        mid_value: float,
    ) -> "DivergingPalette":
        return dataclasses.replace(self, mid_value=mid_value)


## } MODULE
