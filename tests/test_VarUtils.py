## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
from Loki.WWLogging import VarUtils


## ###############################################################
## TESTS
## ###############################################################
class TestVarUtils(unittest.TestCase):
  def test_correct_type_single(self):
    """Typical case: variable is of the required type."""
    VarUtils.assertType(42, int)
  
  def test_correct_type_multiple(self):
    """Typical case: variable matches one of the multiple required types."""
    VarUtils.assertType("Hello", (str, int))

  def test_incorrect_type(self):
    """Edge case 1: variable is not of the required type."""
    with self.assertRaises(TypeError):
      VarUtils.assertType(42, str)

  def test_incorrect_type_multiple(self):
    """Edge case 2: variable doesn't match any of the required types."""
    with self.assertRaises(TypeError):
      VarUtils.assertType(42.5, (str, int))

  def test_list_as_type(self):
    """Edge case 3: list passed as the required type(s)."""
    VarUtils.assertType("Hello", [str, int])
    with self.assertRaises(TypeError):
      VarUtils.assertType(42.5, [str, int])

  def test_single_type_tuple(self):
    """Edge case 4: single type passed in a tuple."""
    VarUtils.assertType(42, (int,))
    with self.assertRaises(TypeError):
      VarUtils.assertType("string", (int,))

  def test_no_type_check(self):
    """Edge case 5: empty tuple means no type check."""
    ## now expecting a `ValueError` instead of silently passing with an empty tuple
    with self.assertRaises(ValueError):
      VarUtils.assertType(42, ())

  def test_variable_name_in_error_message(self):
    """Edge case 6: variable name should be included in error message."""
    ## expect "unknown variable" in the error message if `var_name` is not provided
    with self.assertRaises(TypeError) as cm:
      VarUtils.assertType(42, str)
    self.assertIn("unknown variable", str(cm.exception))
    self.assertIn("str", str(cm.exception))
    self.assertIn("int", str(cm.exception))

  def test_variable_name_in_error_message_with_var_name(self):
    """Edge case 7: variable name should be included in the error message when passed."""
    ## expect the provided `var_name` ("test_var") in the error message
    with self.assertRaises(TypeError) as cm:
      VarUtils.assertType(42, str, "test_var")
    self.assertIn("test_var", str(cm.exception))
    self.assertIn("str", str(cm.exception))
    self.assertIn("int", str(cm.exception))


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST
