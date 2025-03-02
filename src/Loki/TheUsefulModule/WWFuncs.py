## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import time, warnings, inspect


## ###############################################################
## WORKING WITH FUNCTIONS
## ###############################################################
def time_function(func):
  def wrapper(*args, **kwargs):
    time_start = time.time()
    result = func(*args, **kwargs)
    time_elapsed = time.time() - time_start
    print(f"{func.__name__}() took {time_elapsed:.3f} seconds to execute.")
    return result
  return wrapper

def warn_if_result_is_unused(func):
  def wrapper(*args, **kwargs):
    result = func(*args, **kwargs)
    calling_frame = inspect.currentframe().f_back
    ## check if the result is being assigned
    src = inspect.getsource(calling_frame)
    call_line = src.split("\n")[calling_frame.f_lineno - calling_frame.f_code.co_firstlineno]
    if not any([
        elem in call_line
        for elem in ["=", "return"]
      ]):
      warnings.warn(f"Return value of {func.__name__} is not being used", UserWarning, stacklevel=2)
    return result
  return wrapper


## END OF LIBRARY