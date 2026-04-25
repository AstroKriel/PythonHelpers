## { U-TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest
from typing import Any

## third-party
import numpy

## local
from jormi.ww_arrays import mask_2d_arrays
from jormi.ww_types import box_positions

##
## === TEST SUITES
##


class TestGet2dShape(unittest.TestCase):

    def test_returns_shape(
        self,
    ):
        array = numpy.zeros((4, 7))
        self.assertEqual(
            mask_2d_arrays.get_2d_shape(array),
            (4, 7),
        )

    def test_rejects_1d(
        self,
    ):
        with self.assertRaises(Exception):
            mask_2d_arrays.get_2d_shape(numpy.zeros(5))

    def test_rejects_3d(
        self,
    ):
        with self.assertRaises(Exception):
            mask_2d_arrays.get_2d_shape(numpy.zeros((2, 3, 4)))


class TestHalfMasks2D(unittest.TestCase):

    def _check_mask_inverse(
        self,
        *,
        num_rows: int,
        num_cols: int,
        side_a: box_positions.Positions.Side,
        side_b: box_positions.Positions.Side,
    ) -> None:
        ## build two opposing half masks
        mask_a = mask_2d_arrays.HalfMasks2D.get_mask(
            num_rows=num_rows,
            num_cols=num_cols,
            anchor=side_a,
        )
        mask_b = mask_2d_arrays.HalfMasks2D.get_mask(
            num_rows=num_rows,
            num_cols=num_cols,
            anchor=side_b,
        )
        ## the two masks must be exact bitwise inverses of each other
        self.assertTrue(
            numpy.array_equal(mask_a, ~mask_b),
        )
        ## and together they must cover every pixel
        self.assertTrue(
            numpy.all(mask_a | mask_b),
        )

    def test_top_bottom_complement_odd(
        self,
    ):
        self._check_mask_inverse(
            num_rows=5,
            num_cols=6,
            side_a=box_positions.Positions.Side.Top,
            side_b=box_positions.Positions.Side.Bottom,
        )

    def test_top_bottom_complement_even(
        self,
    ):
        self._check_mask_inverse(
            num_rows=4,
            num_cols=6,
            side_a=box_positions.Positions.Side.Top,
            side_b=box_positions.Positions.Side.Bottom,
        )

    def test_left_right_complement_odd(
        self,
    ):
        self._check_mask_inverse(
            num_rows=5,
            num_cols=5,
            side_a=box_positions.Positions.Side.Left,
            side_b=box_positions.Positions.Side.Right,
        )

    def test_left_right_complement_even(
        self,
    ):
        self._check_mask_inverse(
            num_rows=4,
            num_cols=6,
            side_a=box_positions.Positions.Side.Left,
            side_b=box_positions.Positions.Side.Right,
        )

    def test_top_covers_correct_rows(
        self,
    ):
        ## 5 rows: midpoint = (5-1) // 2 = 2, so Top covers rows 0-2
        mask = mask_2d_arrays.HalfMasks2D.get_mask(
            num_rows=5,
            num_cols=4,
            anchor=box_positions.Positions.Side.Top,
        )
        self.assertTrue(
            numpy.all(mask[:3, :]),
        )
        self.assertTrue(
            numpy.all(~mask[3:, :]),
        )

    def test_left_covers_correct_cols(
        self,
    ):
        ## 6 cols: midpoint = (6-1) // 2 = 2, so Left covers cols 0-2
        mask = mask_2d_arrays.HalfMasks2D.get_mask(
            num_rows=4,
            num_cols=6,
            anchor=box_positions.Positions.Side.Left,
        )
        self.assertTrue(
            numpy.all(mask[:, :3]),
        )
        self.assertTrue(
            numpy.all(~mask[:, 3:]),
        )

    def test_output_shape(
        self,
    ):
        mask = mask_2d_arrays.HalfMasks2D.get_mask(
            num_rows=3,
            num_cols=7,
            anchor=box_positions.Positions.Side.Right,
        )
        self.assertEqual(
            mask.shape,
            (3, 7),
        )


class TestQuadrantMasks2D(unittest.TestCase):

    def _all_quadrants(
        self,
        *,
        num_rows: int,
        num_cols: int,
    ) -> list[numpy.ndarray[Any, numpy.dtype[Any]]]:
        return [
            mask_2d_arrays.QuadrantMasks2D.get_mask(
                num_rows=num_rows,
                num_cols=num_cols,
                anchor=corner,
            ) for corner in (
                box_positions.Positions.Corner.TopLeft,
                box_positions.Positions.Corner.TopRight,
                box_positions.Positions.Corner.BottomLeft,
                box_positions.Positions.Corner.BottomRight,
            )
        ]

    def test_union_is_full_grid(
        self,
    ):
        masks = self._all_quadrants(
            num_rows=5,
            num_cols=6,
        )
        union = masks[0]
        for mask in masks[1:]:
            union = union | mask
        self.assertTrue(
            numpy.all(union),
        )

    def test_pairwise_disjoint(
        self,
    ):
        masks = self._all_quadrants(
            num_rows=5,
            num_cols=6,
        )
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                self.assertFalse(
                    numpy.any(masks[i] & masks[j]),
                )

    def test_top_left_position(
        self,
    ):
        ## top-left quadrant should include (0, 0) and exclude (4, 5)
        mask = mask_2d_arrays.QuadrantMasks2D.get_mask(
            num_rows=5,
            num_cols=6,
            anchor=box_positions.Positions.Corner.TopLeft,
        )
        self.assertTrue(
            mask[0, 0],
        )
        self.assertFalse(
            mask[4, 5],
        )

    def test_output_shape(
        self,
    ):
        mask = mask_2d_arrays.QuadrantMasks2D.get_mask(
            num_rows=4,
            num_cols=7,
            anchor=box_positions.Positions.Corner.BottomRight,
        )
        self.assertEqual(
            mask.shape,
            (4, 7),
        )


class TestDiagonalMasks2D(unittest.TestCase):

    def test_main_diagonal_complement(
        self,
    ):
        mask_above = mask_2d_arrays.DiagonalMasks2D.get_mask_above_main_diagonal(
            num_rows=5,
            num_cols=5,
        )
        mask_below = mask_2d_arrays.DiagonalMasks2D.get_mask_below_main_diagonal(
            num_rows=5,
            num_cols=5,
        )
        self.assertTrue(
            numpy.array_equal(mask_above, ~mask_below),
        )
        self.assertTrue(
            numpy.all(mask_above | mask_below),
        )

    def test_anti_diagonal_complement(
        self,
    ):
        mask_above = mask_2d_arrays.DiagonalMasks2D.get_mask_above_anti_diagonal(
            num_rows=5,
            num_cols=5,
        )
        mask_below = mask_2d_arrays.DiagonalMasks2D.get_mask_below_anti_diagonal(
            num_rows=5,
            num_cols=5,
        )
        self.assertTrue(
            numpy.array_equal(mask_above, ~mask_below),
        )
        self.assertTrue(
            numpy.all(mask_above | mask_below),
        )

    def test_main_diagonal_spot_check(
        self,
    ):
        ## above_main: row_index <= col_index
        mask_above = mask_2d_arrays.DiagonalMasks2D.get_mask_above_main_diagonal(
            num_rows=4,
            num_cols=4,
        )
        self.assertTrue(
            mask_above[0, 3],
        )  # 0 <= 3
        self.assertTrue(
            mask_above[2, 2],
        )  # 2 <= 2 (on diagonal)
        self.assertFalse(
            mask_above[3, 0],
        )  # 3 > 0

    def test_anti_diagonal_spot_check(
        self,
    ):
        ## above_anti: row_index <= (num_cols - 1) - col_index
        ## for 4x4: row <= 3 - col
        mask_above = mask_2d_arrays.DiagonalMasks2D.get_mask_above_anti_diagonal(
            num_rows=4,
            num_cols=4,
        )
        self.assertTrue(
            mask_above[0, 0],
        )  # 0 <= 3
        self.assertFalse(
            mask_above[3, 3],
        )  # 3 > 0


class TestWedgeMasks2D(unittest.TestCase):

    def test_all_four_wedges_cover_full_grid(
        self,
    ):
        ## each pixel belongs to at least one wedge
        mask_top = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Top,
        )
        mask_bottom = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Bottom,
        )
        mask_left = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Left,
        )
        mask_right = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Right,
        )
        self.assertTrue(
            numpy.all(mask_top | mask_bottom | mask_left | mask_right),
        )

    def test_top_wedge_includes_top_center(
        self,
    ):
        mask = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Top,
        )
        self.assertTrue(
            mask[0, 2],
        )  # top-center pixel

    def test_bottom_wedge_includes_bottom_center(
        self,
    ):
        mask = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=5,
            num_cols=5,
            anchor=box_positions.Positions.Side.Bottom,
        )
        self.assertTrue(
            mask[4, 2],
        )  # bottom-center pixel

    def test_output_shape(
        self,
    ):
        mask = mask_2d_arrays.WedgeMasks2D.get_mask(
            num_rows=4,
            num_cols=6,
            anchor=box_positions.Positions.Side.Top,
        )
        self.assertEqual(
            mask.shape,
            (4, 6),
        )


class TestCircleMasks2D(unittest.TestCase):

    def test_inside_outside_no_overlap_with_boundary(
        self,
    ):
        mask_inside = mask_2d_arrays.CircleMasks2D.get_mask_inside(
            num_rows=7,
            num_cols=7,
            include_boundary=True,
        )
        mask_outside = mask_2d_arrays.CircleMasks2D.get_mask_outside(
            num_rows=7,
            num_cols=7,
            include_boundary=True,
        )
        self.assertFalse(
            numpy.any(mask_inside & mask_outside),
        )

    def test_center_pixel_always_inside(
        self,
    ):
        mask = mask_2d_arrays.CircleMasks2D.get_mask_inside(
            num_rows=7,
            num_cols=7,
        )
        self.assertTrue(
            mask[3, 3],
        )

    def test_custom_radius_zero_covers_only_center(
        self,
    ):
        mask = mask_2d_arrays.CircleMasks2D.get_mask_inside(
            num_rows=5,
            num_cols=5,
            center_row_col=(2.0, 2.0),
            radius=0.0,
            include_boundary=True,
        )
        self.assertTrue(
            mask[2, 2],
        )
        ## all other pixels should be outside
        mask[2, 2] = False
        self.assertFalse(
            numpy.any(mask),
        )

    def test_custom_radius_spot_check(
        self,
    ):
        ## 7x7, center=(3,3), radius=2: (3,5) is on the boundary (distance=2), (3,6) is outside (distance=3)
        mask = mask_2d_arrays.CircleMasks2D.get_mask_inside(
            num_rows=7,
            num_cols=7,
            center_row_col=(3.0, 3.0),
            radius=2.0,
            include_boundary=True,
        )
        self.assertTrue(
            mask[3, 5],
        )  # distance == radius, included
        self.assertFalse(
            mask[3, 6],
        )  # distance > radius, excluded

    def test_off_center_spot_check(
        self,
    ):
        ## 7x7, center=(0,0), radius=1.5: (0,1) is inside (distance=1), (3,3) is outside (dist approx. 4.24)
        mask = mask_2d_arrays.CircleMasks2D.get_mask_inside(
            num_rows=7,
            num_cols=7,
            center_row_col=(0.0, 0.0),
            radius=1.5,
        )
        self.assertTrue(
            mask[0, 1],
        )  # distance = 1 < radius
        self.assertFalse(
            mask[3, 3],
        )  # distance approx. 4.24 > radius

    def test_output_shape(
        self,
    ):
        mask = mask_2d_arrays.CircleMasks2D.get_mask_inside(
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

## } U-TEST
