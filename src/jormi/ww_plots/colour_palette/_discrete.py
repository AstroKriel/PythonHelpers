## { MODULE

##
## === DEPENDENCIES
##

import numpy
import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_plots.colour_palette._base import (
    ColourPalette,
    _get_base_palette,
    _subset_palette,
)

##
## === CLASS
##


@dataclass(frozen=True, kw_only=True)
class DiscretePalette(ColourPalette):
    """
    A discrete colour palette defined by explicit bin boundaries.

    Parameters
    ----------
    boundaries:
        Ordered sequence of bin edges in data space. N boundaries define N-1 bins.
    palette_name:
        Name of the colour palette to sample from. Can be a Matplotlib name,
        a cmasher name, or one of the jormi built-in names.
    palette_range:
        Portion of the palette to use, as a (min, max) tuple in [0, 1].
    colours:
        Optional list of colours to build a custom palette from. If provided,
        palette_name is used only as a label.
    """
    boundaries: tuple[float, ...]
    palette_name: str = "cmr.arctic"
    palette_range: tuple[float, float] = (0.0, 1.0)
    colours: tuple[str, ...] | None = None

    @classmethod
    def from_colours(
        cls,
        *,
        boundaries: tuple[float, ...],
        colours: list[str],
        palette_name: str = "custom",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        return cls(
            boundaries=boundaries,
            colours=tuple(colours),
            palette_name=palette_name,
            palette_range=palette_range,
        )

    @classmethod
    def uniform(
        cls,
        *,
        value_range: tuple[float, float],
        n_bins: int,
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
        colours: list[str] | None = None,
    ) -> "DiscretePalette":
        """Construct with evenly spaced boundaries across value_range."""
        vmin, vmax = value_range
        boundaries = tuple(float(v) for v in numpy.linspace(vmin, vmax, n_bins + 1))
        return cls(
            boundaries=boundaries,
            palette_name=palette_name,
            palette_range=palette_range,
            colours=tuple(colours) if colours is not None else None,
        )

    @property
    def value_range(self) -> tuple[float, float]:
        return (self.boundaries[0], self.boundaries[-1])

    @property
    def _mpl_norm(self) -> mpl_colors.BoundaryNorm:
        n_bins = len(self.boundaries) - 1
        return mpl_colors.BoundaryNorm(
            boundaries=list(self.boundaries),
            ncolors=n_bins,
        )

    @property
    def _mpl_colormap(self) -> mpl_colors.ListedColormap:
        n_bins = len(self.boundaries) - 1
        if self.colours is not None:
            base = mpl_colors.LinearSegmentedColormap.from_list(
                name=self.palette_name,
                colors=list(self.colours),
                N=256,
            )
        else:
            base = _get_base_palette(self.palette_name)
        continuous = _subset_palette(
            palette=base,
            palette_range=self.palette_range,
            name=self.palette_name,
        )
        sampled = continuous(numpy.linspace(0.0, 1.0, n_bins))
        return mpl_colors.ListedColormap(sampled)

    def with_boundaries(
        self,
        boundaries: tuple[float, ...],
    ) -> "DiscretePalette":
        return dataclasses.replace(self, boundaries=boundaries)


## } MODULE
