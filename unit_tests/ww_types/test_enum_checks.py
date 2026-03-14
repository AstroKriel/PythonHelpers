## { TEST

##
## === DEPENDENCIES
##

import unittest
from enum import Enum

from jormi.ww_types import check_enums

##
## === TEST ENUMS
##


class Corners(Enum):
    TopLeft = "upper left"
    TopRight = "upper right"
    BottomLeft = "lower left"
    BottomRight = "lower right"


class Sides(Enum):
    Left = "left"
    Right = "right"


##
## === TEST SUITES
##


class TestEnsureSequenceOfEnums(unittest.TestCase):

    def test_accepts_tuple_and_list(self):
        check_enums.ensure_sequence_of_enums(
            param=(Corners, ),
            param_name="valid_enums",
        )
        check_enums.ensure_sequence_of_enums(
            param=[Corners, Sides],
            param_name="valid_enums",
        )

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            check_enums.ensure_sequence_of_enums(
                param=(),
                param_name="valid_enums",
            )
        with self.assertRaises(ValueError):
            check_enums.ensure_sequence_of_enums(
                param=[],
                param_name="valid_enums",
            )

    def test_rejects_non_sequence(self):
        with self.assertRaises(TypeError):
            check_enums.ensure_sequence_of_enums(
                param=Corners, # type: ignore[arg-type]
                param_name="valid_enums",
            )

    def test_rejects_non_types(self):
        with self.assertRaises(TypeError):
            check_enums.ensure_sequence_of_enums(
                param=["Corners"], # type: ignore[arg-type]
                param_name="valid_enums",
            )
        with self.assertRaises(TypeError):
            check_enums.ensure_sequence_of_enums(
                param=[Corners, "Sides"], # type: ignore[arg-type]
                param_name="valid_enums",
            )

    def test_rejects_non_enum_types(self):
        with self.assertRaises(TypeError):
            check_enums.ensure_sequence_of_enums(
                param=[Corners, int], # type: ignore[arg-type]
                param_name="valid_enums",
            )
        with self.assertRaises(TypeError):
            check_enums.ensure_sequence_of_enums(
                param=(str,), # type: ignore[arg-type]
                param_name="valid_enums",
            )


class TestResolveMember(unittest.TestCase):

    def test_accepts_enum_member(self):
        out = check_enums.resolve_member(
            member=Corners.TopLeft,
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)

    def test_rejects_enum_member_from_wrong_enum(self):
        with self.assertRaises(ValueError):
            check_enums.resolve_member(
                member=Sides.Left,
                valid_enums=Corners,
            )

    def test_accepts_name_case_insensitive(self):
        out = check_enums.resolve_member(
            member="TopLeft",
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)
        out = check_enums.resolve_member(
            member="topleft",
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)
        out = check_enums.resolve_member(
            member=" TOPLEFT ",
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)

    def test_accepts_value_case_insensitive(self):
        out = check_enums.resolve_member(
            member="upper left",
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)
        out = check_enums.resolve_member(
            member=" UPPER LEFT ",
            valid_enums=Corners,
        )
        self.assertIs(out, Corners.TopLeft)

    def test_with_multiple_valid_enums(self):
        out = check_enums.resolve_member(
            member="Left",
            valid_enums=(Corners, Sides),
        )
        self.assertIs(out, Sides.Left)
        out = check_enums.resolve_member(
            member="TopRight",
            valid_enums=(Corners, Sides),
        )
        self.assertIs(out, Corners.TopRight)

    def test_rejects_non_string_non_enum(self):
        with self.assertRaises(TypeError):
            check_enums.resolve_member(
                member=123, # type: ignore[arg-type]
                valid_enums=Corners,
            )
        with self.assertRaises(TypeError):
            check_enums.resolve_member(
                member=None, # type: ignore[arg-type]
                valid_enums=Corners,
            )

    def test_rejects_unknown_string(self):
        with self.assertRaises(ValueError):
            check_enums.resolve_member(
                member="NotAThing",
                valid_enums=Corners,
            )

    def test_ambiguous_across_enums(self):

        class A(Enum):
            X = "shared"

        class B(Enum):
            X = "shared"

        with self.assertRaises(ValueError):
            check_enums.resolve_member(
                member="shared",
                valid_enums=(A, B),
            )
        with self.assertRaises(ValueError):
            check_enums.resolve_member(
                member="X",
                valid_enums=(A, B),
            )


class TestEnsureValidMember(unittest.TestCase):

    def test_passes_and_fails(self):
        check_enums.ensure_valid_member(
            member="TopLeft",
            valid_enums=Corners,
            param_name="loc",
        )
        with self.assertRaises(ValueError):
            check_enums.ensure_valid_member(
                member="Nope",
                valid_enums=Corners,
                param_name="loc",
            )
        with self.assertRaises(TypeError):
            check_enums.ensure_valid_member(
                member=123, # type: ignore[arg-type]
                valid_enums=Corners,
                param_name="loc",
            )


class TestEnsureMemberIn(unittest.TestCase):

    def test_accepts_member_and_string(self):
        valid = (Corners.TopLeft, Corners.TopRight)
        check_enums.ensure_member_in(
            member=Corners.TopLeft,
            valid_members=valid,
            param_name="corner",
        )
        check_enums.ensure_member_in(
            member="TopRight",
            valid_members=valid,
            param_name="corner",
        )

    def test_rejects_outside_subset(self):
        valid = (Corners.TopLeft, Corners.TopRight)
        with self.assertRaises(ValueError):
            check_enums.ensure_member_in(
                member="BottomLeft",
                valid_members=valid,
                param_name="corner",
            )

    def test_rejects_invalid_valid_members_inputs(self):
        with self.assertRaises(ValueError):
            check_enums.ensure_member_in(
                member="TopLeft",
                valid_members=(),
                param_name="corner",
            )
        with self.assertRaises(TypeError):
            check_enums.ensure_member_in(
                member="TopLeft",
                valid_members=[Corners], # type: ignore[arg-type]
                param_name="corner",
            )
        with self.assertRaises(TypeError):
            check_enums.ensure_member_in(
                member="TopLeft",
                valid_members=["TopLeft"], # type: ignore[arg-type]
                param_name="corner",
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    unittest.main()

## } TEST
