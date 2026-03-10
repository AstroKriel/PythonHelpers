## { MODULE

##
## === DEPENDENCIES
##

import numpy
import dataclasses
import matplotlib.colors as mpl_colors

from dataclasses import dataclass

from jormi.ww_plots.color_palette import _base_palette

##
## === CLASS
##


@dataclass(frozen=True, kw_only=True)
class DiscretePalette(_base_palette.ColorPalette):
    """
    A discrete color palette defined by explicit bin boundaries.

    Parameters
    ----------
    boundaries:
        Ordered sequence of bin edges in data space. N boundaries define N-1 bins.
    palette_range:
        Portion of the palette to use, as a (min, max) tuple in [0, 1].
    _base_colormap:
        Internal: the pre-built base colormap. Use from_name, from_colors, or uniform.
    """
    boundaries: tuple[float, ...]
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
        boundaries: tuple[float, ...],
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        return cls(
            boundaries=boundaries,
            palette_range=palette_range,
            _base_colormap=_base_palette.resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        boundaries: tuple[float, ...],
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        base = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom",
            colors=colors,
            N=256,
        )
        return cls(
            boundaries=boundaries,
            palette_range=palette_range,
            _base_colormap=base,
        )

    @classmethod
    def uniform(
        cls,
        *,
        value_range: tuple[float, float],
        n_bins: int,
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        """Construct with evenly spaced boundaries across value_range."""
        vmin, vmax = value_range
        boundaries = tuple(float(v) for v in numpy.linspace(
            start=vmin,
            stop=vmax,
            num=n_bins + 1,
        ))
        return cls.from_name(
            boundaries=boundaries,
            palette_name=palette_name,
            palette_range=palette_range,
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
        continuous = _base_palette.subset_palette(
            palette=self._base_colormap,
            palette_range=self.palette_range,
            name="discrete",
        )
        sampled = continuous(
            numpy.linspace(
                start=0.0,
                stop=1.0,
                num=n_bins,
            ),
        )
        return mpl_colors.ListedColormap(sampled)

    def with_boundaries(
        self,
        boundaries: tuple[float, ...],
    ) -> "DiscretePalette":
        return dataclasses.replace(self, boundaries=boundaries)


## } MODULE
