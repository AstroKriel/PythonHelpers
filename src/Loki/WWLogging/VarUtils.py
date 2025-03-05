## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
from typing import Any


## ###############################################################
## FUNCTIONS
## ###############################################################
def assertType(
    var: Any,
    required_types: type | tuple[type, ...],
    var_name: str = "<not provided>"
  ) -> None:
  """Asserts that a variable is of a specific type(s)."""
  if not isinstance(var, required_types):
    if not isinstance(required_types, type): required_types = (required_types,)
    type_names = ", ".join(
      req_type.__name__
      for req_type in required_types
    )
    raise TypeError(f"Error: Variable `{var_name}` is of type `{type(var).__name__}` instead of `{type_names}`.")


## END OF MODULE