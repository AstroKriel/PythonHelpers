## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from typing import Any, TypeAlias, cast

## third-party
import numpy
from numpy.typing import NDArray

## local
from jormi.ww_validation import validate_arrays, validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === DATA TYPES
##

Mask2D: TypeAlias = NDArray[numpy.bool_]

##
## === ARRAY-LEVEL HELPERS
##


def get_2d_shape(
    array_2d: NDArray[Any],
) -> tuple[int, int]:
    """Validate that array_2d is a 2D array and return it shape: (num_rows, num_cols)."""
    validate_arrays.ensure_dims(
        array=array_2d,
        num_dims=2,
        param_name="array_2d",
    )
    return cast(tuple[int, int], array_2d.shape)


##
## === HELPERS
##


def _get_grid_indices(
    *,
    num_rows: int,
    num_cols: int,
) -> tuple[NDArray[numpy.int_], NDArray[numpy.int_]]:
    """Return (row_indices, col_indices) for a 2D grid."""
    validate_types.ensure_finite_int(
        param=num_rows,
        param_name="num_rows",
        require_positive=True,
        allow_zero=False,
    )
    validate_types.ensure_finite_int(
        param=num_cols,
        param_name="num_cols",
        require_positive=True,
        allow_zero=False,
    )
    indices = numpy.indices((
        num_rows,
        num_cols,
    ))
    return indices[0], indices[1]


##
## === HALF MASKS
##


class HalfMasks2D:
    """Masks that split a 2D grid into halves, selected by side."""

    @staticmethod
    def get_mask(
        *,
        num_rows: int,
        num_cols: int,
        anchor: box_positions.Positions.PositionLike,
    ) -> Mask2D:
        anchor_side = validate_box_positions.as_box_side(anchor)
        BoxSide = box_positions.Positions.Side
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        if anchor_side is BoxSide.Top:
            return row_indices <= (num_rows - 1) // 2
        if anchor_side is BoxSide.Left:
            return col_indices <= (num_cols - 1) // 2
        if anchor_side is BoxSide.Right:
            return col_indices > (num_cols - 1) // 2
        if anchor_side is BoxSide.Bottom:
            return row_indices > (num_rows - 1) // 2
        raise ValueError(
            f"HalfMasks2D.get_mask: anchor must be a Box Side, got {anchor_side!r}.",
        )


##
## === QUADRANTS
##


class QuadrantMasks2D:
    """Masks that select quadrants, keyed by corner position."""

    @staticmethod
    def get_mask(
        *,
        num_rows: int,
        num_cols: int,
        anchor: box_positions.Positions.PositionLike,
    ) -> Mask2D:
        anchor_corner = validate_box_positions.as_box_corner(anchor)
        BoxCorner = box_positions.Positions.Corner
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        mask_top = row_indices <= (num_rows - 1) // 2
        mask_left = col_indices <= (num_cols - 1) // 2
        if anchor_corner is BoxCorner.TopLeft:
            return mask_top & mask_left
        if anchor_corner is BoxCorner.TopRight:
            return mask_top & ~mask_left
        if anchor_corner is BoxCorner.BottomLeft:
            return ~mask_top & mask_left
        if anchor_corner is BoxCorner.BottomRight:
            return ~mask_top & ~mask_left
        raise ValueError(
            f"QuadrantMasks2D.get_mask: unrecognised Box Corner {anchor_corner!r}.",
        )


##
## === DIAGONALS AND WEDGES
##


class DiagonalMasks2D:
    """Masks defined relative to the main and anti-diagonals."""

    @staticmethod
    def get_mask_above_main_diagonal(
        *,
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        return row_indices <= col_indices

    @staticmethod
    def get_mask_below_main_diagonal(
        *,
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        return ~DiagonalMasks2D.get_mask_above_main_diagonal(
            num_rows=num_rows,
            num_cols=num_cols,
        )

    @staticmethod
    def get_mask_above_anti_diagonal(
        *,
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        """Above means: row_indices <= reflected_col_indices."""
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        return row_indices <= (num_cols - 1) - col_indices

    @staticmethod
    def get_mask_below_anti_diagonal(
        *,
        num_rows: int,
        num_cols: int,
    ) -> Mask2D:
        return ~DiagonalMasks2D.get_mask_above_anti_diagonal(
            num_rows=num_rows,
            num_cols=num_cols,
        )


class WedgeMasks2D:
    """Wedge-like regions bounded by both main and anti-diagonals (the X-shape), selected by side."""

    @staticmethod
    def get_mask(
        *,
        num_rows: int,
        num_cols: int,
        anchor: box_positions.Positions.PositionLike,
    ) -> Mask2D:
        anchor_side = validate_box_positions.as_box_side(anchor)
        BoxSide = box_positions.Positions.Side
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        reflected_col_indices = (num_cols - 1) - col_indices
        is_above_main_diagonal = row_indices <= col_indices
        is_below_main_diagonal = row_indices >= col_indices
        is_above_anti_diagonal = row_indices <= reflected_col_indices
        is_below_anti_diagonal = row_indices >= reflected_col_indices
        if anchor_side is BoxSide.Top:
            return is_above_main_diagonal & is_above_anti_diagonal
        if anchor_side is BoxSide.Left:
            return is_above_main_diagonal & is_below_anti_diagonal
        if anchor_side is BoxSide.Right:
            return is_below_main_diagonal & is_above_anti_diagonal
        if anchor_side is BoxSide.Bottom:
            return is_below_main_diagonal & is_below_anti_diagonal
        raise ValueError(
            f"WedgeMasks2D.get_mask: anchor must be a Box Side, got {anchor_side!r}.",
        )


##
## === CIRCULAR REGIONS
##


class CircleMasks2D:
    """Masks for circular regions (and their complements)."""

    @staticmethod
    def get_mask_inside(
        *,
        num_rows: int,
        num_cols: int,
        center_row_col: tuple[float, float] | None = None,
        radius: float | None = None,
        include_boundary: bool = True,
    ) -> Mask2D:
        row_indices, col_indices = _get_grid_indices(
            num_rows=num_rows,
            num_cols=num_cols,
        )
        if center_row_col is None:
            center_row_col = (
                (num_rows - 1) / 2.0,
                (num_cols - 1) / 2.0,
            )
        if radius is None:
            radius = 0.5 * min(num_rows, num_cols)
        r_sq = (row_indices - center_row_col[0])**2 + (col_indices - center_row_col[1])**2
        r_limit_sq = radius * radius
        return r_sq <= r_limit_sq if include_boundary else r_sq < r_limit_sq

    @staticmethod
    def get_mask_outside(
        *,
        num_rows: int,
        num_cols: int,
        center_row_col: tuple[float, float] | None = None,
        radius: float | None = None,
        include_boundary: bool = False,
    ) -> Mask2D:
        return ~CircleMasks2D.get_mask_inside(
            num_rows=num_rows,
            num_cols=num_cols,
            center_row_col=center_row_col,
            radius=radius,
            include_boundary=not (include_boundary),
        )


## } MODULE
