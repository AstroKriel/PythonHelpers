## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import time
import warnings
from typing import Any, Callable

## local
from jormi.ww_io import manage_log

##
## === FUNCTION DECORATORS
##


def time_fn(
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator to measure and log a function's execution time."""

    def wrapper(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        start_time = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Error occurred in {fn.__name__}() while measuring the elapsed time.",
            ) from error
        elapsed_time = time.time() - start_time
        manage_log.log_note(f"{fn.__name__}() took {elapsed_time:.3f}s.")
        return result

    return wrapper


class WarnIfUnused:
    """Wrapper to warn if a wrapped result is not used by the caller."""

    def __init__(
        self,
        *,
        result: Any,
        fn_name: str,
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
        name: str,
    ) -> Any:
        self._result_was_used = True
        return getattr(self._result, name)

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._result_was_used = True
        return self._result(*args, **kwargs)

    def __repr__(
        self,
    ) -> str:
        self._result_was_used = True
        return repr(self._result)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        self._result_was_used = True
        if isinstance(other, WarnIfUnused):
            return self._result == other._result
        return self._result == other

    def __ne__(
        self,
        other: object,
    ) -> bool:
        return not self.__eq__(other)

    def __hash__(
        self,
    ) -> int:
        self._result_was_used = True
        return hash(self._result)

    def __getitem__(
        self,
        key: Any,
    ) -> Any:
        self._result_was_used = True
        return self._result[key]

    def __setitem__(
        self,
        key: Any,
        value: Any,
    ) -> None:
        self._result_was_used = True
        self._result[key] = value

    def unwrap(
        self,
    ) -> Any:
        """Mark the result as used and return the underlying value."""
        self._result_was_used = True
        return self._result


def warn_if_fn_result_is_unused(
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator to warn when a non-None result is ignored by the caller."""

    def wrapper(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = fn(*args, **kwargs)
        if result is not None:
            return WarnIfUnused(
                result=result,
                fn_name=fn.__name__,
            )
        return result

    return wrapper


## } MODULE
