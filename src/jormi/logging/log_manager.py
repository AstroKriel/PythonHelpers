## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################

from enum import Enum
from typing import Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box


## ###############################################################
## SETUP
## ###############################################################

_CONSOLE = Console(highlight=False, soft_wrap=False)

class Level(str, Enum):
  SUCCESS = "success"
  WARNING = "warning"
  ERROR   = "error"
  INFO    = "info"
  LIST    = "list"
  DEBUG   = "debug"
  SKIP    = "skip"

class Status(str, Enum):
  SUCCESS = "success"
  FAILURE = "failure"
  ERROR   = "error"
  WARNING = "warning"
  SKIP    = "skip"

class Colours(str, Enum):
  GREEN  = "#32CD32"
  YELLOW = "#FFE600"
  ORANGE = "#E48500"
  RED    = "#FF4500"
  PURPLE = "#C000EB"
  BLUE   = "#2A71F6"
  WHITE  = "#FFFFFF"
  GREY   = "#818181"
  BLACK  = "#020202"

class Symbols(str, Enum):
  CLOSED_CIRCLE = "\u25CF" # ●
  OPEN_CIRCLE   = "\u25CB" # ○
  ARROW         = "\u21AA" # ↪
  TAG           = "\u003E" # >
  DASH          = "\u2014" # —

LEVEL_ICON: dict[Level, Tuple[str, str]] = {
  Level.INFO    : (Symbols.ARROW.value,         Colours.WHITE.value),
  Level.SUCCESS : (Symbols.CLOSED_CIRCLE.value, Colours.GREEN.value),
  Level.WARNING : (Symbols.CLOSED_CIRCLE.value, Colours.YELLOW.value),
  Level.ERROR   : (Symbols.CLOSED_CIRCLE.value, Colours.RED.value),
  Level.LIST    : (Symbols.TAG.value,           Colours.BLUE.value),
  Level.SKIP    : (Symbols.OPEN_CIRCLE.value,   Colours.GREY.value),
  Level.DEBUG   : (Symbols.OPEN_CIRCLE.value,   Colours.PURPLE.value),
}

STATUS_LABEL: dict[Status, Tuple[str, str]] = {
  Status.SUCCESS : ("SUCCESS", Colours.GREEN.value),
  Status.FAILURE : ("FAILURE", Colours.ORANGE.value),
  Status.ERROR   : ("ERROR",   Colours.RED.value),
  Status.WARNING : ("WARNING", Colours.YELLOW.value),
  Status.SKIP    : ("SKIP",    Colours.GREY.value),
}

@dataclass
class RecordAction:
  action : str
  status : Status = Status.SUCCESS
  status_message : str | None = None
  timestamp : str | None = None
  notes : dict[str, Any] | None = None


## ###############################################################
## FUNCTIONS
## ###############################################################

def get_timestamp() -> str:
  return datetime.now().isoformat(sep=" ", timespec="seconds")

def print_line(
  message     : str,
  level       : Level = Level.INFO,
  timestamp   : str | None = None,
  show_time   : bool = False,
  show_symbol : bool = True,
) -> None:
  ## generate timestamp if not provided
  if timestamp is None: timestamp = get_timestamp()
  ## collect parts of the line: "[icon] [timestamp] message"
  parts: list[str] = []
  ## prepend coloured symbol if enabled
  if show_symbol:
    symbol, color_style = LEVEL_ICON[level]
    parts.append(f"[{color_style}]{symbol}[/]")
  ## prepend timestamp if enabled
  if show_time: parts.append(f"[{timestamp}]")
  ## always append main message
  parts.append(message)
  ## join parts with spaces and render via Rich
  _CONSOLE.print(" ".join(parts))

def print_block(
  record    : RecordAction,
  show_time : bool = True,
  min_width : int = 50,
  max_width : int = 100,
) -> None:
  status_text, status_color = STATUS_LABEL[record.status]
  ## include timestamp prefix if enabled
  timestamp_prefix = f"[{record.timestamp or get_timestamp()}]" if show_time else ""
  ## build panel title: "[time] Action : STATUS"
  title_text = Text()
  if timestamp_prefix:
    title_text.append(timestamp_prefix, style=Colours.GREY.value)
    title_text.append(" ")
  title_text.append(record.action)
  title_text.append(" : ")
  title_text.append(status_text, style=status_color)
  ## build panel body lines from notes (if provided)
  row_prefix = Symbols.DASH.value
  body_lines: list[Text] = []
  if record.notes:
    if not isinstance(record.notes, dict):
      raise TypeError("RecordAction.notes must be a dict[str, Any] if provided.")
    for key_label, value in record.notes.items():
      line = Text(f"{row_prefix} ", style=Colours.GREY.value)
      line.append(str(key_label), style=Colours.GREY.value)
      line.append(" : ")
      line.append(str(value), style="bold")
      body_lines.append(line)
  ## add trailing status_message
  if record.status_message:
    body_lines.append(Text(f"{row_prefix} {record.status_message}", style=status_color))
  ## compute width needed to fit content
  content_width = title_text.cell_len
  if body_lines:
    content_width = max(content_width, max(t.cell_len for t in body_lines))
  ## add padding + clamp width to [min_width, max_width]
  horizontal_padding = 2
  borders = 2
  panel_width_needed = content_width + horizontal_padding + borders
  panel_width = max(min_width, min(panel_width_needed, max_width))
  ## render text block
  panel = Panel(
    Text("\n").join(body_lines),
    title       = title_text,
    title_align = "left",
    box         = box.ROUNDED,
    width       = panel_width,
    padding     = (0, 1),
  )
  _CONSOLE.print(panel)
  _CONSOLE.print()


## END OF MODULE