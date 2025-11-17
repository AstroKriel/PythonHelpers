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
    """Decorator to measure and log a function's execution time."""

    def wrapper(
        *args,
        **kwargs,
    ):
        start_time = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Error occurred in {fn.__name__}() while measuring the elapsed time.",
            ) from error
        elapsed_time = time.time() - start_time
        print(f"{fn.__name__}() took {elapsed_time:.3f} seconds to execute.")
        return result

    return wrapper


class WarnIfUnused:
    """Wrapper to warn if a wrapped result is not used by the caller."""

    def __init__(
        self,
        result,
        fn_name,
    ):
        self._result = result
        self._fn_name = fn_name
        self._result_was_used = False

    def __del__(
        self,
    ):
        if not self._result_was_used:
            warnings.warn(
                message=f"The result returned by {self._fn_name}() was never used.",
                category=UserWarning,
                stacklevel=2,
            )

    def __getattr__(
        self,
        name,
    ):
        self._result_was_used = True
        return getattr(self._result, name)

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        self._result_was_used = True
        return self._result(*args, **kwargs)

    def __repr__(
        self,
    ):
        self._result_was_used = True
        return repr(self._result)


def warn_if_fn_result_is_unused(
    fn,
):
    """Decorator to warn when a non-None result is ignored by the caller."""

    def wrapper(
        *args,
        **kwargs,
    ):
        result = fn(*args, **kwargs)
        if result is not None:
            return WarnIfUnused(
                result=result,
                fn_name=fn.__name__,
            )
        return result

    return wrapper


## } MODULE
