## { MODULE

##
## === DEPENDENCIES
##

from enum import Enum
from typing import Mapping, Any, Literal, Iterable
from datetime import datetime
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

##
## === TYPES + CONSTANTS
## enums, dataclasses, and symbols/colours used across the api
##

_CONSOLE = Console(highlight=False, soft_wrap=False)


class _Colours(str, Enum):
    GREEN = "#32CD32"
    YELLOW = "#FFE600"
    ORANGE = "#E48500"
    RED = "#FF4500"
    PURPLE = "#C000EB"
    BLUE = "#2A71F6"
    LIGHTBLUE = "#7ED1E8"
    WHITE = "#FFFFFF"
    GREY = "#818181"
    BLACK = "#020202"


class Symbols(str, Enum):
    CLOSED_CIRCLE = "\u25CF"  # ●
    OPEN_CIRCLE = "\u25CB"  # ○
    RIGHT_ARROW = "\u2192"  # →
    HOOKED_ARROW = "\u21AA"  # ↪
    GREATER_THAN = "\u003E"  # >
    EM_DASH = "\u2014"  # —


@dataclass(frozen=True)
class _MessageStyle:
    message: str
    icon: str
    colour: str


class MessageType(Enum):
    TASK = _MessageStyle("Task", Symbols.RIGHT_ARROW.value, _Colours.WHITE.value)
    NOTE = _MessageStyle("Note", Symbols.HOOKED_ARROW.value, _Colours.WHITE.value)
    ACTION = _MessageStyle("Action", Symbols.CLOSED_CIRCLE.value, _Colours.WHITE.value)
    HINT = _MessageStyle("Hint", Symbols.CLOSED_CIRCLE.value, _Colours.YELLOW.value)
    ALERT = _MessageStyle("Alert", Symbols.CLOSED_CIRCLE.value, _Colours.ORANGE.value)
    DEBUG = _MessageStyle("Debug", Symbols.OPEN_CIRCLE.value, _Colours.PURPLE.value)
    LIST = _MessageStyle("List", Symbols.EM_DASH.value, _Colours.BLUE.value)
    SUMMARY = _MessageStyle("Summary", Symbols.GREATER_THAN.value, _Colours.LIGHTBLUE.value)
    SECTION = _MessageStyle("Section", Symbols.GREATER_THAN.value, _Colours.WHITE.value)

    def requires_outcome(self) -> bool:
        return self is MessageType.ACTION


class ActionOutcome(Enum):
    SUCCESS = _MessageStyle("Success", Symbols.CLOSED_CIRCLE.value, _Colours.GREEN.value)
    FAILURE = _MessageStyle("Failure", Symbols.CLOSED_CIRCLE.value, _Colours.RED.value)
    ERROR = _MessageStyle("Error", Symbols.CLOSED_CIRCLE.value, _Colours.RED.value)
    WARNING = _MessageStyle("Warning", Symbols.CLOSED_CIRCLE.value, _Colours.YELLOW.value)
    SKIPPED = _MessageStyle("Skipped", Symbols.OPEN_CIRCLE.value, _Colours.ORANGE.value)


##
## === MESSAGE MODEL
## data container for messages used by the renderers and log-helpers
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
                raise ValueError(
                    "MessageType.ACTION requires 'action_outcome' to be set.",
                )
            return self.action_outcome.value
        else:
            if self.action_outcome is not None:
                raise ValueError(
                    "'action_outcome' should not be set when Message.message_type != MessageType.ACTION.",
                )
            return self.message_type.value


##
## === HELPERS
##


def get_timestamp() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


##
## === RENDERING API
## internal api used by the log-helpers
##


def render_blank(
    *,
    count: int = 1,
) -> None:
    if count < 1:
        return
    for _ in range(count):
        _CONSOLE.print()


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
        line_parts.append(f"[{_Colours.GREY.value}][{timestamp}][/]")
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
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    ## validate option
    if message_position not in ("top", "bottom"):
        raise ValueError("render_block(message_position): expected 'top' or 'bottom'.")
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
    ## build body lines from optional message + notes
    row_prefix = Symbols.EM_DASH.value
    body_lines: list[Text] = []
    ## optional message at top
    if message.message and message_position == "top":
        body_lines.append(Text(f"{row_prefix} {message.message}", style=message_style.colour))
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
    ## optional message at bottom (default)
    if message.message and message_position == "bottom":
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
    if add_spacing:
        _CONSOLE.print()


##
## === SINGLE LINE LOGGING
##


def log_empty_lines(*, lines: int = 1) -> None:
    render_blank(count=lines)


def log_task(
    text: str,
    *,
    show_time: bool = True,
) -> None:
    render_line(
        Message(text, message_type=MessageType.TASK),
        show_time=show_time,
    )


def log_note(
    text: str,
    *,
    show_time: bool = True,
) -> None:
    render_line(
        Message(text, message_type=MessageType.NOTE),
        show_time=show_time,
    )


def log_hint(
    text: str,
    *,
    show_time: bool = True,
) -> None:
    render_line(
        Message(text, message_type=MessageType.HINT),
        show_time=show_time,
    )


def log_alert(
    text: str,
    *,
    show_time: bool = True,
) -> None:
    render_line(
        Message(text, message_type=MessageType.ALERT),
        show_time=show_time,
    )


def log_debug(
    text: str,
    *,
    show_time: bool = True,
) -> None:
    render_line(
        Message(text, message_type=MessageType.DEBUG),
        show_time=show_time,
    )


def log_outcome(
    text: str,
    *,
    outcome: ActionOutcome,
    show_time: bool = True,
) -> None:
    render_line(
        Message(
            text,
            message_type=MessageType.ACTION,
            action_outcome=outcome,
        ),
        show_time=show_time,
    )


##
## === LOGGING W/ GROUPED INFO
##


def log_action(
    *,
    title: str,
    succeeded: bool | None,
    message: str = "",
    notes: Mapping[str, object] | None = None,
    show_time: bool = True,
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    if succeeded is None:
        outcome = ActionOutcome.SKIPPED
    elif succeeded:
        outcome = ActionOutcome.SUCCESS
    else:
        outcome = ActionOutcome.FAILURE
    message_notes: dict[str, Any] = dict(notes) if notes else {}
    render_block(
        Message(
            message=message,
            message_type=MessageType.ACTION,
            message_title=title,
            action_outcome=outcome,
            message_notes=message_notes,
        ),
        show_time=show_time,
        message_position=message_position,
    )


def log_context(
    *,
    title: str,
    message: str = "",
    notes: Mapping[str, object] | None = None,
    show_time: bool = True,
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    message_notes: dict[str, Any] = dict(notes) if notes else {}
    render_block(
        Message(
            message=message,
            message_type=MessageType.NOTE,
            message_title=title,
            message_notes=message_notes,
        ),
        show_time=show_time,
        message_position=message_position,
    )


def log_items(
    *,
    title: str,
    items: Iterable[Any],
    message: str = "",
    show_time: bool = True,
    message_position: Literal["top", "bottom"] = "top",
) -> None:
    grouped_items: dict[str, Any] = {f"{i+1}": item for i, item in enumerate(items)}
    render_block(
        Message(
            message=message,
            message_type=MessageType.LIST,
            message_title=title,
            message_notes=grouped_items,
        ),
        show_time=show_time,
        message_position=message_position,
    )


def log_warning(
    text: str,
    notes: Mapping[str, object] | None = None,
    *,
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    message_notes: dict[str, Any] = dict(notes) if notes else {}
    render_block(
        Message(
            message=text,
            message_type=MessageType.ACTION,
            message_title="Warning",
            action_outcome=ActionOutcome.WARNING,
            message_notes=message_notes,
        ),
        show_time=True,
        message_position=message_position,
    )


def log_error(
    text: str,
    notes: Mapping[str, object] | None = None,
    *,
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    message_notes: dict[str, Any] = dict(notes) if notes else {}
    render_block(
        Message(
            message=text,
            message_type=MessageType.ACTION,
            message_title="Error",
            action_outcome=ActionOutcome.ERROR,
            message_notes=message_notes,
        ),
        show_time=True,
        message_position=message_position,
    )


def log_summary(
    *,
    title: str,
    notes: Mapping[str, object],
    message: str = "",
    show_time: bool = True,
    message_position: Literal["top", "bottom"] = "bottom",
) -> None:
    message_notes: dict[str, Any] = dict(notes)
    render_block(
        Message(
            message=message,
            message_type=MessageType.SUMMARY,
            message_title=title,
            message_notes=message_notes,
        ),
        show_time=show_time,
        message_position=message_position,
    )


def log_section(
    title: str,
    *,
    show_time: bool = False,
    add_spacing: bool = False,
) -> None:
    render_line(
        Message(title, message_type=MessageType.SECTION),
        show_time=show_time,
        add_spacing=add_spacing,
    )


## } MODULE
