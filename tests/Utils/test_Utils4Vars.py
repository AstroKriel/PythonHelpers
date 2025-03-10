## ###############################################################
## DEPENDENCIES
## ###############################################################
import unittest
from Loki.Utils import Utils4Vars


## ###############################################################
## TESTS
## ###############################################################
class Test_Utils4Vars_assertType(unittest.TestCase):
  def test_correct_type_single(self):
    ## typical case: variable is of the required type.
    Utils4Vars.assertType(42, int)
  
  def test_correct_type_multiple(self):
    ## typical case: variable matches one of the multiple required types.
    Utils4Vars.assertType("Hello", (str, int))

  def test_incorrect_type(self):
    ## edge case 1: variable is not of the required type.
    with self.assertRaises(TypeError):
      Utils4Vars.assertType(42, str)

  def test_incorrect_type_multiple(self):
    ## edge case 2: variable does not match any of the required types.
    with self.assertRaises(TypeError):
      Utils4Vars.assertType(42.5, (str, int))

  def test_list_as_type(self):
    ## edge case 3: list passed as the required type(s).
    Utils4Vars.assertType("Hello", [str, int])
    with self.assertRaises(TypeError):
      Utils4Vars.assertType(42.5, [str, int])

  def test_single_type_tuple(self):
    ## edge case 4: single type passed in a tuple.
    Utils4Vars.assertType(42, (int,))
    with self.assertRaises(TypeError):
      Utils4Vars.assertType("string", (int,))

  def test_no_type_check(self):
    ## edge case 5: empty tuple means no type check.
    ## now expecting a `ValueError` instead of silently passing with an empty tuple
    with self.assertRaises(ValueError):
      Utils4Vars.assertType(42, ())

  def test_variable_name_in_error_message(self):
    ## edge case 6: variable name should be included in error message.
    ## expect "unknown variable" in the error message if `var_name` is not provided
    with self.assertRaises(TypeError) as cm:
      Utils4Vars.assertType(42, str)
    self.assertIn("unknown variable", str(cm.exception))
    self.assertIn("str", str(cm.exception))
    self.assertIn("int", str(cm.exception))

  def test_variable_name_in_error_message_with_var_name(self):
    ## edge case 7: variable name should be included in the error message when passed.
    ## expect the provided `var_name` ("test_var") in the error message
    with self.assertRaises(TypeError) as cm:
      Utils4Vars.assertType(42, str, "test_var")
    self.assertIn("test_var", str(cm.exception))
    self.assertIn("str", str(cm.exception))
    self.assertIn("int", str(cm.exception))


## ###############################################################
## TEST ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  unittest.main()


## END OF TEST