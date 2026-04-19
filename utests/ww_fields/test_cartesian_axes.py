## { TEST

##
## === DEPENDENCIES
##

## stdlib
import unittest

## local
from jormi.ww_fields import cartesian_axes

##
## === TEST SUITES
##


class TestCartesianAxis_3D(unittest.TestCase):

    def test_round_trip_invariants(
        self,
    ):
        for axis_index, axis in enumerate(cartesian_axes.DEFAULT_3D_AXES_ORDER):
            self.assertIs(
                cartesian_axes.as_axis(axis=axis),
                axis,
            )
            self.assertIs(
                cartesian_axes.as_axis(axis=axis.axis_label),
                axis,
            )
            self.assertIs(
                cartesian_axes.as_axis(axis=axis.axis_index),
                axis,
            )
            self.assertEqual(
                cartesian_axes.get_axis_index(axis=axis),
                axis_index,
            )
            self.assertEqual(
                cartesian_axes.get_axis_label(axis=axis),
                f"x_{axis_index}",
            )

    def test_enum_member_values(
        self,
    ):
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X0.value,
            "x_0",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X1.value,
            "x_1",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X2.value,
            "x_2",
        )

    def test_enum_member_properties(
        self,
    ):
        ## axis labels
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X0.axis_label,
            "x_0",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X1.axis_label,
            "x_1",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X2.axis_label,
            "x_2",
        )
        ## axis index
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X0.axis_index,
            0,
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X1.axis_index,
            1,
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X2.axis_index,
            2,
        )

    def test_default_3d_axes(
        self,
    ):
        self.assertEqual(
            cartesian_axes.VALID_3D_AXIS_LABELS,
            ("x_0", "x_1", "x_2"),
        )
        self.assertEqual(
            cartesian_axes.VALID_3D_AXIS_INDICES,
            (0, 1, 2),
        )
        self.assertEqual(
            cartesian_axes.DEFAULT_3D_AXES_ORDER,
            (
                cartesian_axes.CartesianAxis_3D.X0,
                cartesian_axes.CartesianAxis_3D.X1,
                cartesian_axes.CartesianAxis_3D.X2,
            ),
        )


class TestAsAxis(unittest.TestCase):

    def test_accepts_enum_member(
        self,
    ):
        out_axis = cartesian_axes.as_axis(axis=cartesian_axes.CartesianAxis_3D.X1)
        self.assertIs(out_axis, cartesian_axes.CartesianAxis_3D.X1)

    def test_accepts_index(
        self,
    ):
        self.assertIs(
            cartesian_axes.as_axis(axis=0),
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis=1),
            cartesian_axes.CartesianAxis_3D.X1,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis=2),
            cartesian_axes.CartesianAxis_3D.X2,
        )

    def test_rejects_invalid_index(
        self,
    ):
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis=-1)
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis=3)

    def test_accepts_string_case_insensitive(
        self,
    ):
        ## resolve by value
        self.assertIs(
            cartesian_axes.as_axis(axis="x_0"),
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis=" x_1 "),
            cartesian_axes.CartesianAxis_3D.X1,
        )
        ## resolve by member name (backwards-compatible)
        self.assertIs(
            cartesian_axes.as_axis(axis="X0"),
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis="x0"),
            cartesian_axes.CartesianAxis_3D.X0,
        )

    def test_rejects_unknown_string(
        self,
    ):
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis="1")
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis="x3")
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis="bla")

    def test_rejects_invalid_type(
        self,
    ):
        with self.assertRaises(TypeError):
            cartesian_axes.as_axis(axis=None)  # type: ignore
        with self.assertRaises(TypeError):
            cartesian_axes.as_axis(axis=1.0)  # type: ignore


class TestGetAxisIndexAndLabel(unittest.TestCase):

    def test_getters_accept_all_inputs(
        self,
    ):
        self.assertEqual(
            cartesian_axes.get_axis_label(axis=cartesian_axes.CartesianAxis_3D.X2),
            "x_2",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index(axis=cartesian_axes.CartesianAxis_3D.X2),
            2,
        )

        self.assertEqual(
            cartesian_axes.get_axis_label(axis="X0"),
            "x_0",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index(axis="X0"),
            0,
        )

        self.assertEqual(
            cartesian_axes.get_axis_label(axis=1),
            "x_1",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index(axis=1),
            1,
        )

    def test_getters_raise_on_invalid(
        self,
    ):
        with self.assertRaises(ValueError):
            cartesian_axes.get_axis_label(axis="x3")
        with self.assertRaises(ValueError):
            cartesian_axes.get_axis_index(axis=3)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
