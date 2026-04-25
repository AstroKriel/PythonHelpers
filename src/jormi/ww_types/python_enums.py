## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from enum import Enum

##
## === TYPES
##

EnumType = type[Enum]
EnumTypesLike = EnumType | tuple[EnumType, ...] | list[EnumType]
EnumMemberLike = Enum | str

## } MODULE
