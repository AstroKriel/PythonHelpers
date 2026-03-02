## { TEST

##
## === DEPENDENCIES
##

import unittest

from jormi.ww_fields import cartesian_axes

##
## === TEST SUITES
##


class TestCartesianAxis_3D(unittest.TestCase):

    def test_enum_member_values(self):
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X0.value,
            "x0",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X1.value,
            "x1",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X2.value,
            "x2",
        )

    def test_enum_member_properties(self):
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X0.axis_label,
            "x0",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X1.axis_label,
            "x1",
        )
        self.assertEqual(
            cartesian_axes.CartesianAxis_3D.X2.axis_label,
            "x2",
        )

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

    def test_valid_constants(self):
        self.assertEqual(
            cartesian_axes.VALID_3D_AXIS_LABELS,
            ("x0", "x1", "x2"),
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

    def test_accepts_enum_member(self):
        out = cartesian_axes.as_axis(axis=cartesian_axes.CartesianAxis_3D.X1)
        self.assertIs(out, cartesian_axes.CartesianAxis_3D.X1)

    def test_accepts_index(self):
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

    def test_rejects_invalid_index(self):
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis=-1)
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis=3)

    def test_accepts_string_case_insensitive(self):
        self.assertIs(
            cartesian_axes.as_axis(axis="x0"),
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis="X0"),
            cartesian_axes.CartesianAxis_3D.X0,
        )
        self.assertIs(
            cartesian_axes.as_axis(axis=" x1 "),
            cartesian_axes.CartesianAxis_3D.X1,
        )

    def test_rejects_unknown_string(self):
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis="x3")
        with self.assertRaises(ValueError):
            cartesian_axes.as_axis(axis="nope")

    def test_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            cartesian_axes.as_axis(axis=None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cartesian_axes.as_axis(axis=1.0)  # type: ignore[arg-type]


class TestGetAxisIndexAndLabel(unittest.TestCase):

    def test_getters_accept_all_inputs(self):
        self.assertEqual(
            cartesian_axes.get_axis_label(
                cartesian_axes.CartesianAxis_3D.X2,
            ),
            "x2",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index(
                cartesian_axes.CartesianAxis_3D.X2,
            ),
            2,
        )

        self.assertEqual(
            cartesian_axes.get_axis_label("X0"),
            "x0",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index("X0"),
            0,
        )

        self.assertEqual(
            cartesian_axes.get_axis_label(1),
            "x1",
        )
        self.assertEqual(
            cartesian_axes.get_axis_index(1),
            1,
        )

    def test_getters_raise_on_invalid(self):
        with self.assertRaises(ValueError):
            cartesian_axes.get_axis_label("x3")
        with self.assertRaises(ValueError):
            cartesian_axes.get_axis_index(3)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
