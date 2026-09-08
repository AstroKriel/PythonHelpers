## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import dataclasses

## local
from jormi.ww_validation import validate_types

##
## === LATEX LABEL
##


@dataclasses.dataclass(frozen=True)
class LatexLabel:
    """Raw LaTeX math-mode content, without the surrounding `$` delimiters.

    Combine labels by embedding `.content` into a new `LatexLabel`, then read `.label`
    once, at the point a string actually needs to be handed to a plot.
    """

    content: str

    def __post_init__(
        self,
    ) -> None:
        validate_types.ensure_nonempty_string(
            param=self.content,
            param_name="<content>",
        )
        if "$" in self.content:
            raise ValueError(
                f"`<content>` must not include `$`; read `.label` once instead of embedding it, got: {self.content!r}",
            )
        if self.content.count("{") != self.content.count("}"):
            raise ValueError(f"`<content>` has unbalanced braces: {self.content!r}")

    @property
    def label(
        self,
    ) -> str:
        return f"${self.content}$"


## } MODULE
