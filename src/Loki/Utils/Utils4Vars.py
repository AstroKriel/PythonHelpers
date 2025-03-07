## START OF MODULE


## ###############################################################
## FUNCTIONS
## ###############################################################
def assertType(
    var,
    tuple_required_types: type | tuple[type, ...],
    var_name: str = None
  ) -> None:
  """Asserts that a variable is of a specific type(s)."""
  if not tuple_required_types: raise ValueError("Error: no required types were passed.")
  if var_name is None: var_name = "<unknown variable name>"
  ## isinstance() only accepts a single instance or a tuple, because tuples are immutable
  if   isinstance(tuple_required_types, type): tuple_required_types = (tuple_required_types,)
  elif isinstance(tuple_required_types, list): tuple_required_types = tuple(tuple_required_types)
  if not isinstance(var, tuple_required_types):
    type_names = ", ".join(
      req_type.__name__
      for req_type in tuple_required_types
    )
    raise TypeError(f"Error: Variable `{var_name}` is of type `{type(var).__name__}` instead of `{type_names}`.")


## END OF MODULE