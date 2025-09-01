## { MODULE

##
## === DEPENDENCIES ===
##

from enum import Enum
from typing import Any
from datetime import datetime
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

##
## === SETUP ===
##

_CONSOLE = Console(highlight=False, soft_wrap=False)


class _Colours(str, Enum):
    GREEN = "#32CD32"
    YELLOW = "#FFE600"
    ORANGE = "#E48500"
    RED = "#FF4500"
    PURPLE = "#C000EB"
    BLUE = "#2A71F6"
    WHITE = "#FFFFFF"
    GREY = "#818181"
    BLACK = "#020202"


class _Symbols(str, Enum):
    CLOSED_CIRCLE = "\u25CF"  # ●
    OPEN_CIRCLE = "\u25CB"  # ○
    HOOKED_ARROW = "\u21AA"  # ↪
    GREATER_THAN = "\u003E"  # >
    EM_DASH = "\u2014"  # —


@dataclass(frozen=True)
class _MessageStyle:
    message: str
    icon: str
    colour: str


class MessageType(Enum):
    GENERAL = _MessageStyle("GENERAL", _Symbols.HOOKED_ARROW.value, _Colours.WHITE.value)
    ACTION = _MessageStyle("ACTION", _Symbols.CLOSED_CIRCLE.value, _Colours.WHITE.value)
    HINT = _MessageStyle("HINT", _Symbols.CLOSED_CIRCLE.value, _Colours.YELLOW.value)
    ALERT = _MessageStyle("ALERT", _Symbols.CLOSED_CIRCLE.value, _Colours.ORANGE.value)
    LIST = _MessageStyle("LIST", _Symbols.GREATER_THAN.value, _Colours.BLUE.value)
    DEBUG = _MessageStyle("DEBUG", _Symbols.OPEN_CIRCLE.value, _Colours.PURPLE.value)

    def requires_outcome(self) -> bool:
        return self is MessageType.ACTION


class ActionOutcome(Enum):
    SUCCESS = _MessageStyle("SUCCESS", _Symbols.CLOSED_CIRCLE.value, _Colours.GREEN.value)
    FAILURE = _MessageStyle("FAILURE", _Symbols.CLOSED_CIRCLE.value, _Colours.RED.value)
    ERROR = _MessageStyle("ERROR", _Symbols.CLOSED_CIRCLE.value, _Colours.RED.value)
    WARNING = _MessageStyle("WARNING", _Symbols.CLOSED_CIRCLE.value, _Colours.YELLOW.value)
    SKIPPED = _MessageStyle("SKIPPED", _Symbols.OPEN_CIRCLE.value, _Colours.GREY.value)


##
## === MESSAGE CONTAINER ===
##


@dataclass
class Message:
    message: str
    message_type: MessageType
    message_title: str | None = None
    action_outcome: ActionOutcome | None = None
    message_notes: dict[str, Any] | None = None
    timestamp: str | None = None

    def style(self) -> _MessageStyle:
        ## validate and return the appropriate style payload
        ## note, it only makes sense to have an outcome associated with actions
        if self.message_type.requires_outcome():
            if self.action_outcome is None:
                raise ValueError("MessageType.ACTION requires 'action_outcome' to be set.")
            return self.action_outcome.value
        else:
            if self.action_outcome is not None:
                raise ValueError(
                    "'action_outcome' should not be set when Message.message_type != MessageType.ACTION.",
                )
            return self.message_type.value


##
## === HELPERS ===
##


def get_timestamp() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


##
## === RENDERING MESSAGES ===
##


def render_line(
    message: Message,
    *,
    show_time: bool = False,
    show_symbol: bool = True,
    add_spacing: bool = False,
) -> None:
    ## ensure line-only constraints (no titles on lines)
    if message.message_title is not None:
        raise ValueError("'message_title' is only valid for blocks; omit it when rendering a line.")
    ## collect style and timestamp for this line
    timestamp = message.timestamp or get_timestamp()
    message_style = message.style()
    line_parts: list[str] = []
    ## prepend coloured symbol if enabled
    if show_symbol:
        line_parts.append(f"[{message_style.colour}]{message_style.icon}[/]")
    ## prepend timestamp if enabled
    if show_time:
        line_parts.append(f"[{timestamp}]")
    ## prepend outcome label for ACTION lines
    if message.message_type is MessageType.ACTION:
        line_parts.append(message_style.message + ":")
    ## always append the main message body
    line_parts.append(message.message)
    ## join parts with spaces and render
    _CONSOLE.print(" ".join(line_parts))
    ## add spacing line if requested
    if add_spacing:
        _CONSOLE.print()


def render_block(
    message: Message,
    *,
    show_time: bool = True,
    min_width: int = 60,
    max_width: int = 100,
    add_spacing: bool = True,
) -> None:
    ## collect style and timestamp for this block
    message_style = message.style()
    timestamp_prefix = f"[{message.timestamp or get_timestamp()}]" if show_time else ""
    ## build panel title: "[time] Title : STATUS/LABEL"
    panel_title = Text()
    if timestamp_prefix:
        panel_title.append(timestamp_prefix, style=_Colours.GREY.value)
        panel_title.append(" ")
    panel_title.append(message.message_title or "Untitled")
    panel_title.append(" : ")
    panel_title.append(message_style.message, style=message_style.colour)
    ## build body lines from notes and optional trailing message
    row_prefix = _Symbols.EM_DASH.value
    body_lines: list[Text] = []
    ## include notes as "— key : value" entries
    if message.message_notes:
        if not isinstance(message.message_notes, dict):
            raise TypeError("Message.message_notes must be a dict[str, Any] if provided.")
        for key_label, value in message.message_notes.items():
            note_line = Text(f"{row_prefix} ", style=_Colours.GREY.value)
            note_line.append(str(key_label), style=_Colours.GREY.value)
            note_line.append(" : ")
            note_line.append(str(value), style="bold")
            body_lines.append(note_line)
    ## append final trailing message if present
    if message.message:
        body_lines.append(Text(f"{row_prefix} {message.message}", style=message_style.colour))
    ## compute width required to fit title and body lines
    content_width = panel_title.cell_len
    if body_lines:
        content_width = max(content_width, max(line.cell_len for line in body_lines))
    h_padding = 2
    borders = 2
    panel_width_needed = content_width + h_padding + borders
    panel_width = max(min_width, min(panel_width_needed, max_width))
    ## render panel
    panel = Panel(
        Text("\n").join(body_lines),
        title=panel_title,
        title_align="left",
        box=box.ROUNDED,
        width=panel_width,
        padding=(0, 1),
    )
    _CONSOLE.print(panel)
    ## add spacing line if requested
    if add_spacing:
        _CONSOLE.print()


## } MODULE
