## { MODULE

##
## === DEPENDENCIES
##

## third-party
import numpy

##
## === TYPES
##


class Types:
    """Public namespace for common Python type groupings; use for both annotations and value selection."""

    class Strings:
        StringLike = str

    class Booleans:
        BooleanLike = bool | numpy.bool_

    class Numerics:
        IntLike = int | numpy.integer
        FloatLike = float | numpy.floating
        NumericLike = IntLike | FloatLike

    class Containers:
        SetLike = set
        DictLike = dict
        ArrayLike = numpy.ndarray
        ContainerLike = SetLike | DictLike | ArrayLike

    class Sequences:
        ListLike = list
        TupleLike = tuple
        SequenceLike = ListLike | TupleLike


## } MODULE
