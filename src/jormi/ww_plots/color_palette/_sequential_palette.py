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
class SequentialPalette(ColourPalette):
    """
    A continuous, single-direction colour palette.

    Parameters
    ----------
    value_range:
        Data-space (vmin, vmax) tuple.
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
    palette_name: str = "cmr.arctic"
    palette_range: tuple[float, float] = (0.0, 1.0)
    colours: tuple[str, ...] | None = None

    @classmethod
    def from_colours(
        cls,
        *,
        value_range: tuple[float, float],
        colours: list[str],
        palette_name: str = "custom",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "SequentialPalette":
        return cls(
            value_range=value_range,
            colours=tuple(colours),
            palette_name=palette_name,
            palette_range=palette_range,
        )

    @property
    def _mpl_norm(self) -> mpl_colors.Normalize:
        return _create_norm(
            value_range=self.value_range,
            mid_value=None,
        )

    def with_value_range(
        self,
        value_range: tuple[float, float],
    ) -> "SequentialPalette":
        return dataclasses.replace(self, value_range=value_range)


## } MODULE
