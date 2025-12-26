## { MODULE

##
## === DEPENDENCIES
##

import numpy

from typing import TypeAlias
from numpy.typing import NDArray
from dataclasses import dataclass

from jormi.ww_types import array_checks, type_checks
from jormi.ww_types import cardinal_anchors, ordinal_anchors

##
## === DATA TYPES
##

Mask2D: TypeAlias = NDArray[numpy.bool_]
IndexGrid2D: TypeAlias = NDArray[numpy.int_]

##
## === DATA STRUCTURES
##

@dataclass(frozen=True)
class HalfMasks:
    """Masks that split a 2D grid into halves."""

    top: Mask2D
    bottom: Mask2D
    left: Mask2D
    right: Mask2D


@dataclass(frozen=True)
class CornerMasks:
    """Masks that split a 2D grid into quadrants."""

    top_left: Mask2D
    top_right: Mask2D
    bottom_left: Mask2D
    bottom_right: Mask2D


@dataclass(frozen=True)
class AboveBelowMask:
    """Grouped masks that select regions above/below."""

    above: Mask2D
    below: Mask2D


@dataclass(frozen=True)
class InsideOutsideMask:
    """Grouped masks that select regions inside/outside."""

    inside: Mask2D
    outside: Mask2D


##
## === HELPER FUNCTIONS
##


def _get_grid_indices(
    num_rows: int,
    num_cols: int,
) -> tuple[IndexGrid2D, IndexGrid2D]:
    """Return (row_index, col_index) for a 2D grid."""
    type_checks.ensure_finite_int(
        param=num_rows,
        param_name="num_rows",
        require_positive=True,
        allow_zero=False,
    )
    type_checks.ensure_finite_int(
        param=num_cols,
        param_name="num_cols",
        require_positive=True,
        allow_zero=False,
    )
    indices: NDArray[numpy.int_] = numpy.indices((num_rows, num_cols))
    row_index: IndexGrid2D = indices[0]
    col_index: IndexGrid2D = indices[1]
    return row_index, col_index


def _get_half_masks(
    num_rows: int,
    num_cols: int,
) -> HalfMasks:
    """Return grouped masks for top/bottom/left/right halves."""
    row_index, col_index = _get_grid_indices(num_rows, num_cols)
    mask_left = col_index <= (num_cols - 1) // 2
    mask_right = col_index >= num_cols // 2
    mask_top = row_index <= (num_rows - 1) // 2
    mask_bottom = row_index >= (num_rows - 1) // 2
    return HalfMasks(
        top=mask_top,
        bottom=mask_bottom,
        left=mask_left,
        right=mask_right,
    )


def _get_ordinal_masks(
    num_rows: int,
    num_cols: int,
) -> CornerMasks:
    """Return grouped quadrant masks built from the cardinal halves."""
    halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
    top_left = halves.top & halves.left
    top_right = halves.top & halves.right
    bottom_left = halves.bottom & halves.left
    bottom_right = halves.bottom & halves.right
    return CornerMasks(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
    )


def _get_main_diagonal_masks(
    num_rows: int,
    num_cols: int,
) -> AboveBelowMask:
    """Return masks above/below the main diagonal."""
    row_index, col_index = _get_grid_indices(num_rows, num_cols)
    mask_above = row_index <= col_index
    mask_below = row_index >= col_index
    return AboveBelowMask(
        above=mask_above,
        below=mask_below,
    )


def _get_anti_diagonal_masks(
    num_rows: int,
    num_cols: int,
) -> AboveBelowMask:
    """Return masks above/below the anti-diagonal."""
    row_index, col_index = _get_grid_indices(num_rows, num_cols)
    reflected_col_index = (num_cols - 1) - col_index
    mask_above = row_index <= reflected_col_index
    mask_below = row_index >= reflected_col_index
    return AboveBelowMask(
        above=mask_above,
        below=mask_below,
    )


def _get_circle_mask(
    num_rows: int,
    num_cols: int,
    *,
    center_row_col: tuple[float, float] | None,
    radius: float | None,
    include_boundary: bool,
) -> InsideOutsideMask:
    """
    Return a boolean mask for points inside a circle.

    The coordinates are in (row, col) index space.
    If center_row_col is None, the centre of the grid is used.
    If radius is None, half of the smaller dimension is used.
    """
    row_index, col_index = _get_grid_indices(num_rows, num_cols)
    if center_row_col is None:
        center_row_col = ((num_rows - 1) / 2.0, (num_cols - 1) / 2.0)
    if radius is None:
        radius = 0.5 * min(num_rows, num_cols)
    delta_row = row_index - center_row_col[0]
    delta_col = col_index - center_row_col[1]
    radius_squared = delta_row * delta_row + delta_col * delta_col
    radius_limit_squared = radius * radius
    if include_boundary:
        mask_inside = radius_squared <= radius_limit_squared
    else:
        mask_inside = radius_squared < radius_limit_squared
    return InsideOutsideMask(
        inside=mask_inside,
        outside=~mask_inside,
    )


##
## === USER-FACING: HALF MASKS
##


class HalfMasks2D:
    """
    Masks that split a 2D grid into halves, selected by cardinal anchors.
    """

    @staticmethod
    def get_vertical_half_mask(
        num_rows: int,
        num_cols: int,
        anchor: cardinal_anchors.VerticalAnchorLike,
    ) -> Mask2D:
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        v_anchor = cardinal_anchors.as_vertical_anchor(anchor)
        if v_anchor is cardinal_anchors.VerticalAnchor.Top:
            return halves.top
        if v_anchor is cardinal_anchors.VerticalAnchor.Bottom:
            return halves.bottom
        raise ValueError(
            "HalfMasks2D.get_vertical_half_mask: "
            "anchor must be VerticalAnchor.Top or VerticalAnchor.Bottom.",
        )

    @staticmethod
    def get_horizontal_half_mask(
        num_rows: int,
        num_cols: int,
        anchor: cardinal_anchors.HorizontalAnchorLike,
    ) -> Mask2D:
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        h_anchor = cardinal_anchors.as_horizontal_anchor(anchor)
        if h_anchor is cardinal_anchors.HorizontalAnchor.Left:
            return halves.left
        if h_anchor is cardinal_anchors.HorizontalAnchor.Right:
            return halves.right
        raise ValueError(
            "HalfMasks2D.get_horizontal_half_mask: "
            "anchor must be HorizontalAnchor.Left or HorizontalAnchor.Right.",
        )

    @staticmethod
    def get_mask_top_half(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """Convenience wrapper for the top half."""
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        return halves.top

    @staticmethod
    def get_mask_bottom_half(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """Convenience wrapper for the bottom half."""
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        return halves.bottom

    @staticmethod
    def get_mask_left_half(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """Convenience wrapper for the left half."""
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        return halves.left

    @staticmethod
    def get_mask_right_half(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """Convenience wrapper for the right half."""
        halves = _get_half_masks(num_rows=num_rows, num_cols=num_cols)
        return halves.right


##
## === USER-FACING: QUADRANTS
##


class QuadrantMasks2D:
    """
    Masks that select quadrants using halves, keyed by corner anchors.

    Only the four corner anchors are supported:
    - CornerAnchor.TopLeft
    - CornerAnchor.TopRight
    - CornerAnchor.BottomLeft
    - CornerAnchor.BottomRight
    """

    @staticmethod
    def get_quadrant_mask(
        num_rows: int,
        num_cols: int,
        anchor: ordinal_anchors.CornerAnchorLike,
    ) -> Mask2D:
        quadrants = _get_ordinal_masks(num_rows=num_rows, num_cols=num_cols)
        corner = ordinal_anchors.as_corner_anchor(anchor)
        CornerAnchor = ordinal_anchors.CornerAnchor
        if corner is CornerAnchor.TopLeft:
            return quadrants.top_left
        if corner is CornerAnchor.TopRight:
            return quadrants.top_right
        if corner is CornerAnchor.BottomLeft:
            return quadrants.bottom_left
        if corner is CornerAnchor.BottomRight:
            return quadrants.bottom_right
        raise ValueError(
            "QuadrantMasks2D.get_quadrant_mask: "
            "anchor must be one of "
            "{CornerAnchor.TopLeft, CornerAnchor.TopRight, "
            "CornerAnchor.BottomLeft, CornerAnchor.BottomRight}.",
        )


##
## === USER-FACING: DIAGONALS & WEDGES
##


class DiagonalMasks2D:
    """
    Masks defined relative to the main and anti-diagonals.
    """

    @staticmethod
    def get_mask_above_main_diagonal(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        main_masks = _get_main_diagonal_masks(num_rows, num_cols)
        return main_masks.above

    @staticmethod
    def get_mask_below_main_diagonal(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        main_masks = _get_main_diagonal_masks(num_rows, num_cols)
        return main_masks.below

    @staticmethod
    def get_mask_above_anti_diagonal(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """
        Anti-diagonal uses reflected_col_index = (num_cols - 1) - col_index.
        'Above' means row_index <= reflected_col_index.
        """
        anti_masks = _get_anti_diagonal_masks(num_rows, num_cols)
        return anti_masks.above

    @staticmethod
    def get_mask_below_anti_diagonal(
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        anti_masks = _get_anti_diagonal_masks(num_rows, num_cols)
        return anti_masks.below


class WedgeMasks2D:
    """
    Wedge-like regions bounded by BOTH diagonals (the X-shape).

    Vertical wedges are selected by VerticalAnchor (top/bottom).
    Horizontal wedges are selected by HorizontalAnchor (left/right).
    """

    @staticmethod
    def get_vertical_wedge_mask(
        num_rows: int,
        num_cols: int,
        anchor: cardinal_anchors.VerticalAnchorLike,
    ) -> Mask2D:
        main_masks = _get_main_diagonal_masks(num_rows, num_cols)
        anti_masks = _get_anti_diagonal_masks(num_rows, num_cols)
        v_anchor = cardinal_anchors.as_vertical_anchor(anchor)
        if v_anchor is cardinal_anchors.VerticalAnchor.Top:
            return main_masks.above & anti_masks.above
        if v_anchor is cardinal_anchors.VerticalAnchor.Bottom:
            return main_masks.below & anti_masks.below
        raise ValueError(
            "WedgeMasks2D.get_vertical_wedge_mask: "
            "anchor must be VerticalAnchor.Top or VerticalAnchor.Bottom.",
        )

    @staticmethod
    def get_horizontal_wedge_mask(
        num_rows: int,
        num_cols: int,
        anchor: cardinal_anchors.HorizontalAnchorLike,
    ) -> Mask2D:
        main_masks = _get_main_diagonal_masks(num_rows, num_cols)
        anti_masks = _get_anti_diagonal_masks(num_rows, num_cols)
        h_anchor = cardinal_anchors.as_horizontal_anchor(anchor)
        if h_anchor is cardinal_anchors.HorizontalAnchor.Left:
            return main_masks.above & anti_masks.below
        if h_anchor is cardinal_anchors.HorizontalAnchor.Right:
            return main_masks.below & anti_masks.above
        raise ValueError(
            "WedgeMasks2D.get_horizontal_wedge_mask: "
            "anchor must be HorizontalAnchor.Left or HorizontalAnchor.Right.",
        )


##
## === USER-FACING: CIRCULAR REGIONS
##


class CircleMasks2D:
    """Masks for circular regions (and their complements)."""

    @staticmethod
    def get_mask_inside_circle(
        num_rows: int,
        num_cols: int,
        *,
        center_row_col: tuple[float, float] | None = None,
        radius: float | None = None,
        include_boundary: bool = True,
    ) -> Mask2D:
        circle_mask = _get_circle_mask(
            num_rows,
            num_cols,
            center_row_col=center_row_col,
            radius=radius,
            include_boundary=include_boundary,
        )
        return circle_mask.inside

    @staticmethod
    def get_mask_outside_circle(
        num_rows: int,
        num_cols: int,
        *,
        center_row_col: tuple[float, float] | None = None,
        radius: float | None = None,
        include_boundary: bool = False,
    ) -> Mask2D:
        circle_mask = _get_circle_mask(
            num_rows,
            num_cols,
            center_row_col=center_row_col,
            radius=radius,
            include_boundary=include_boundary,
        )
        return circle_mask.outside


##
## === ARRAY-LEVEL HELPERS
##


def get_vertical_half_mask_for_array(
    array_2d: numpy.ndarray,
    anchor: cardinal_anchors.VerticalAnchorLike,
) -> Mask2D:
    """Convenience wrapper to build a vertical half mask for a 2D array."""
    array_checks.ensure_dims(
        array=array_2d,
        num_dims=2,
        param_name="array_2d",
    )
    num_rows, num_cols = array_2d.shape
    return HalfMasks2D.get_vertical_half_mask(
        num_rows=num_rows,
        num_cols=num_cols,
        anchor=anchor,
    )


def get_horizontal_half_mask_for_array(
    array_2d: numpy.ndarray,
    anchor: cardinal_anchors.HorizontalAnchorLike,
) -> Mask2D:
    """Convenience wrapper to build a horizontal half mask for a 2D array."""
    array_checks.ensure_dims(
        array=array_2d,
        num_dims=2,
        param_name="array_2d",
    )
    num_rows, num_cols = array_2d.shape
    return HalfMasks2D.get_horizontal_half_mask(
        num_rows=num_rows,
        num_cols=num_cols,
        anchor=anchor,
    )


def get_quadrant_mask_for_array(
    array_2d: numpy.ndarray,
    anchor: ordinal_anchors.CornerAnchorLike,
) -> Mask2D:
    """Convenience wrapper to build a quadrant mask for a 2D array."""
    array_checks.ensure_dims(
        array=array_2d,
        num_dims=2,
        param_name="array_2d",
    )
    num_rows, num_cols = array_2d.shape
    return QuadrantMasks2D.get_quadrant_mask(
        num_rows=num_rows,
        num_cols=num_cols,
        anchor=anchor,
    )


## } MODULE
