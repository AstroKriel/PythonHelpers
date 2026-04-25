## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

from dataclasses import dataclass

## third-party
import matplotlib.colors as mpl_colors
import numpy

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
## === DISCRETE PALETTE
##


@dataclass(frozen=True, kw_only=True)
class DiscretePalette(ColorPalette):
    """
    A discrete color palette defined by explicit bin edges.

    Use `from_name`, `from_colors`, or `from_uniform_range` to construct.
    """

    bin_edges: tuple[float, ...]  # strictly increasing bin edges; N edges define N-1 bins
    palette_range: tuple[float, float] = (0.0, 1.0)  # portion of palette to use, in [0, 1]

    def __post_init__(
        self,
    ) -> None:
        validate_python_types.ensure_tuple_of_numbers(
            param=self.bin_edges,
            param_name="bin_edges",
        )
        if len(self.bin_edges) < 2:
            raise ValueError("`bin_edges` must have at least 2 elements.")
        for boundary_idx in range(len(self.bin_edges) - 1):
            if not (float(self.bin_edges[boundary_idx]) < float(self.bin_edges[boundary_idx + 1])):
                raise ValueError(
                    f"`bin_edges` must be strictly increasing;"
                    f" got bin_edges[{boundary_idx}]={self.bin_edges[boundary_idx]}"
                    f" >= bin_edges[{boundary_idx + 1}]={self.bin_edges[boundary_idx + 1]}.",
                )
        ensure_palette_range(self.palette_range)

    @classmethod
    def from_name(
        cls,
        *,
        bin_edges: tuple[float, ...],
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        return cls(
            bin_edges=bin_edges,
            palette_range=palette_range,
            _base_cmap=resolve_palette(palette_name),
        )

    @classmethod
    def from_colors(
        cls,
        *,
        bin_edges: tuple[float, ...],
        colors: list[str],
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        ## look-up table size should be 256 to match 8-bit color depth (this exceeds perceptual resolution)
        cmap = mpl_colors.LinearSegmentedColormap.from_list(
            name="custom-discrete-cmap",
            colors=colors,
            N=256,
        )
        return cls(
            bin_edges=bin_edges,
            palette_range=palette_range,
            _base_cmap=cmap,
        )

    @classmethod
    def from_uniform_range(
        cls,
        *,
        value_range: tuple[float, float],
        num_bins: int,
        palette_name: str = "cmr.arctic",
        palette_range: tuple[float, float] = (0.0, 1.0),
    ) -> "DiscretePalette":
        """Construct with evenly spaced bin_edges across value_range."""
        value_min, value_max = value_range
        bin_edges = tuple(
            float(value) for value in numpy.linspace(
                start=value_min,
                stop=value_max,
                num=num_bins + 1,
            )
        )
        return cls.from_name(
            bin_edges=bin_edges,
            palette_name=palette_name,
            palette_range=palette_range,
        )

    @property
    def value_range(
        self,
    ) -> tuple[float, float]:
        return (
            self.bin_edges[0],
            self.bin_edges[-1],
        )

    @property
    def mpl_norm(
        self,
    ) -> mpl_colors.BoundaryNorm:
        num_bins = len(self.bin_edges) - 1
        return mpl_colors.BoundaryNorm(
            boundaries=list(self.bin_edges),
            ncolors=num_bins,
        )

    @property
    def mpl_cmap(
        self,
    ) -> mpl_colors.ListedColormap:
        num_bins = len(self.bin_edges) - 1
        continuous_cmap = subset_palette(
            palette_cmap=self._base_cmap,
            palette_range=self.palette_range,
            palette_label="subset-discrete-cmap",
        )
        sampled_colors = continuous_cmap(
            numpy.linspace(
                start=0.0,
                stop=1.0,
                num=num_bins,
            ),
        )
        return mpl_colors.ListedColormap(sampled_colors)

    def with_bin_edges(
        self,
        bin_edges: tuple[float, ...],
    ) -> "DiscretePalette":
        return dataclasses.replace(self, bin_edges=bin_edges)


## } MODULE
