## { TEST

##
## === DEPENDENCIES
##

import unittest

import numpy

from jormi.ww_types import box_positions
from jormi.ww_arrays.mask_2d_arrays import (
    get_2d_shape,
    HalfMasks2D,
    QuadrantMasks2D,
    DiagonalMasks2D,
    WedgeMasks2D,
    CircleMasks2D,
)

##
## === SHORTCUTS
##

Side = box_positions.TypeHints.Box.Side
Corner = box_positions.TypeHints.Box.Corner

##
## === TEST SUITES
##


class TestGet2dShape(unittest.TestCase):

    def test_returns_shape(self):
        array = numpy.zeros((4, 7))
        self.assertEqual(
            get_2d_shape(array),
            (4, 7),
        )

    def test_rejects_1d(self):
        with self.assertRaises(Exception):
            get_2d_shape(numpy.zeros(5))

    def test_rejects_3d(self):
        with self.assertRaises(Exception):
            get_2d_shape(numpy.zeros((2, 3, 4)))


class TestHalfMasks2D(unittest.TestCase):

    def _check_complement(
        self,
        num_rows,
        num_cols,
        side_a,
        side_b,
    ):
        mask_a = HalfMasks2D.get_half_mask(
            num_rows=num_rows,
            num_cols=num_cols,
            anchor=side_a,
        )
        mask_b = HalfMasks2D.get_half_mask(
            num_rows=num_rows,
            num_cols=num_cols,
            anchor=side_b,
        )
        self.assertTrue(numpy.array_equal(mask_a, ~mask_b))
        self.assertTrue(numpy.all(mask_a | mask_b))

    def test_top_bottom_complement_odd(self):
        self._check_complement(
            num_rows=5,
            num_cols=6,
            side_a=Side.Top,
            side_b=Side.Bottom,
        )

    def test_top_bottom_complement_even(self):
        self._check_complement(
            num_rows=4,
            num_cols=6,
            side_a=Side.Top,
            side_b=Side.Bottom,
        )

    def test_left_right_complement_odd(self):
        self._check_complement(
            num_rows=5,
            num_cols=5,
            side_a=Side.Left,
            side_b=Side.Right,
        )

    def test_left_right_complement_even(self):
        self._check_complement(
            num_rows=4,
            num_cols=6,
            side_a=Side.Left,
            side_b=Side.Right,
        )

    def test_top_covers_correct_rows(self):
        ## 5 rows: midpoint = (5-1)//2 = 2, so Top covers rows 0-2
        mask = HalfMasks2D.get_half_mask(
            num_rows=5,
            num_cols=4,
            anchor=Side.Top,
        )
        self.assertTrue(numpy.all(mask[:3, :]))
        self.assertTrue(numpy.all(~mask[3:, :]))

    def test_left_covers_correct_cols(self):
        ## 6 cols: midpoint = (6-1)//2 = 2, so Left covers cols 0-2
        mask = HalfMasks2D.get_half_mask(
            num_rows=4,
            num_cols=6,
            anchor=Side.Left,
        )
        self.assertTrue(numpy.all(mask[:, :3]))
        self.assertTrue(numpy.all(~mask[:, 3:]))

    def test_output_shape(self):
        mask = HalfMasks2D.get_half_mask(
            num_rows=3,
            num_cols=7,
            anchor=Side.Right,
        )
        self.assertEqual(
            mask.shape,
            (3, 7),
        )


class TestQuadrantMasks2D(unittest.TestCase):

    def _all_quadrants(
        self,
        num_rows,
        num_cols,
    ):
        return [
            QuadrantMasks2D.get_quadrant_mask(
                num_rows=num_rows,
                num_cols=num_cols,
                anchor=corner,
            ) for corner in (
                Corner.TopLeft,
                Corner.TopRight,
                Corner.BottomLeft,
                Corner.BottomRight,
            )
        ]

    def test_union_is_full_grid(self):
        masks = self._all_quadrants(
            num_rows=5,
            num_cols=6,
        )
        union = masks[0]
        for mask in masks[1:]:
            union = union | mask
        self.assertTrue(numpy.all(union))

    def test_pairwise_disjoint(self):
        masks = self._all_quadrants(
            num_rows=5,
            num_cols=6,
        )
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                self.assertFalse(numpy.any(masks[i] & masks[j]))

    def test_top_left_position(self):
        ## top-left quadrant should include (0, 0) and exclude (4, 5)
        mask = QuadrantMasks2D.get_quadrant_mask(
            num_rows=5,
            num_cols=6,
            anchor=Corner.TopLeft,
        )
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[4, 5])

    def test_output_shape(self):
        mask = QuadrantMasks2D.get_quadrant_mask(
            num_rows=4,
            num_cols=7,
            anchor=Corner.BottomRight,
        )
        self.assertEqual(
            mask.shape,
            (4, 7),
        )


class TestDiagonalMasks2D(unittest.TestCase):

    def test_main_diagonal_complement(self):
        mask_above = DiagonalMasks2D.get_mask_above_main_diagonal(
            num_rows=5,
            num_cols=5,
        )
        mask_below = DiagonalMasks2D.get_mask_below_main_diagonal(
            num_rows=5,
            num_cols=5,
        )
        self.assertTrue(numpy.array_equal(mask_above, ~mask_below))
        self.assertTrue(numpy.all(mask_above | mask_below))

    def test_anti_diagonal_complement(self):
        mask_above = DiagonalMasks2D.get_mask_above_anti_diagonal(
            num_rows=5,
            num_cols=5,
        )
        mask_below = DiagonalMasks2D.get_mask_below_anti_diagonal(
            num_rows=5,
            num_cols=5,
        )
        self.assertTrue(numpy.array_equal(mask_above, ~mask_below))
        self.assertTrue(numpy.all(mask_above | mask_below))

    def test_main_diagonal_spot_check(self):
        ## above_main: row_index <= col_index
        mask_above = DiagonalMasks2D.get_mask_above_main_diagonal(
            num_rows=4,
            num_cols=4,
        )
        self.assertTrue(mask_above[0, 3])  # 0 <= 3
        self.assertTrue(mask_above[2, 2])  # 2 <= 2 (on diagonal)
        self.assertFalse(mask_above[3, 0])  # 3 > 0

    def test_anti_diagonal_spot_check(self):
        ## above_anti: row_index <= (num_cols - 1) - col_index
        ## for 4x4: row <= 3 - col
        mask_above = DiagonalMasks2D.get_mask_above_anti_diagonal(
            num_rows=4,
            num_cols=4,
        )
        self.assertTrue(mask_above[0, 0])  # 0 <= 3
        self.assertFalse(mask_above[3, 3])  # 3 > 0


class TestWedgeMasks2D(unittest.TestCase):

    def test_all_four_wedges_cover_full_grid(self):
        ## each pixel belongs to at least one wedge
        mask_top = WedgeMasks2D.get_vertical_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Top,
        )
        mask_bottom = WedgeMasks2D.get_vertical_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Bottom,
        )
        mask_left = WedgeMasks2D.get_horizontal_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Left,
        )
        mask_right = WedgeMasks2D.get_horizontal_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Right,
        )
        self.assertTrue(numpy.all(mask_top | mask_bottom | mask_left | mask_right))

    def test_top_wedge_includes_top_center(self):
        mask = WedgeMasks2D.get_vertical_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Top,
        )
        self.assertTrue(mask[0, 2])  # top-center pixel

    def test_bottom_wedge_includes_bottom_center(self):
        mask = WedgeMasks2D.get_vertical_wedge_mask(
            num_rows=5,
            num_cols=5,
            anchor=Side.Bottom,
        )
        self.assertTrue(mask[4, 2])  # bottom-center pixel

    def test_output_shape(self):
        mask = WedgeMasks2D.get_vertical_wedge_mask(
            num_rows=4,
            num_cols=6,
            anchor=Side.Top,
        )
        self.assertEqual(
            mask.shape,
            (4, 6),
        )


class TestCircleMasks2D(unittest.TestCase):

    def test_inside_outside_no_overlap_with_boundary(self):
        mask_inside = CircleMasks2D.get_mask_inside_circle(
            num_rows=7,
            num_cols=7,
            include_boundary=True,
        )
        mask_outside = CircleMasks2D.get_mask_outside_circle(
            num_rows=7,
            num_cols=7,
            include_boundary=True,
        )
        self.assertFalse(numpy.any(mask_inside & mask_outside))

    def test_center_pixel_always_inside(self):
        mask = CircleMasks2D.get_mask_inside_circle(
            num_rows=7,
            num_cols=7,
        )
        self.assertTrue(mask[3, 3])

    def test_custom_radius_zero_covers_only_center(self):
        mask = CircleMasks2D.get_mask_inside_circle(
            num_rows=5,
            num_cols=5,
            center_row_col=(2.0, 2.0),
            radius=0.0,
            include_boundary=True,
        )
        self.assertTrue(mask[2, 2])
        ## all other pixels should be outside
        mask[2, 2] = False
        self.assertFalse(numpy.any(mask))

    def test_output_shape(self):
        mask = CircleMasks2D.get_mask_inside_circle(
            num_rows=4,
            num_cols=6,
        )
        self.assertEqual(
            mask.shape,
            (4, 6),
        )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
