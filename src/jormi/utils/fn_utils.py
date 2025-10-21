## { MODULE

##
## === DEPENDENCIES
##

import time
import warnings

##
## === FUNCTION DECORATORS
##


def time_fn(
    fn,
):

    def wrapper(
        *args,
        **kwargs,
    ):
        start_time = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Error occurred in {fn.__name__}() while measuring the elapsed time."
            ) from error
        elapsed_time = time.time() - start_time
        print(f"{fn.__name__}() took {elapsed_time:.3f} seconds to execute.")
        return result

    return wrapper


class WarnIfUnused:

    def __init__(
        self,
        value,
        fn_name,
    ):
        self._value = value
        self._fn_name = fn_name
        self._value_was_used = False

    def __del__(
        self,
    ):
        if not self._value_was_used:
            warnings.warn(
                message=f"The value returned by {self._fn_name}() was never used.",
                category=UserWarning,
                stacklevel=2,
            )

    def __getattr__(
        self,
        name,
    ):
        self._value_was_used = True
        return getattr(self._value, name)

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        self._value_was_used = True
        return self._value(*args, **kwargs)

    def __repr__(
        self,
    ):
        self._value_was_used = True
        return repr(self._value)


def warn_if_fn_result_is_unused(
    fn,
):

    def wrapper(
        *args,
        **kwargs,
    ):
        result = fn(*args, **kwargs)
        if result is not None:
            return WarnIfUnused(result, fn.__name__)
        return result

    return wrapper


## } MODULE
