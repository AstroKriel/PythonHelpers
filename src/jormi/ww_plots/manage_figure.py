## { MODULE

##
## === WORKSPACE SETUP
##

import matplotlib

matplotlib.use("Agg", force=True)

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TypeAlias,
    overload,
)

## third-party
import numpy
from matplotlib import pyplot as mpl_plot
from matplotlib import rcParams
from matplotlib.axes import Axes as mpl_Axes
from matplotlib.figure import Figure as mpl_Figure
from numpy.typing import NDArray

## local
from jormi.ww_io import (
    manage_io,
    manage_log,
    manage_shell,
)
from jormi.ww_plots import style_figure
from jormi.ww_validation import validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === TYPE ALIASES
##

Panel: TypeAlias = mpl_Axes
PanelGrid: TypeAlias = NDArray[numpy.object_]

##
## === BOX SHAPE
##


@dataclass(
    frozen=True,
    kw_only=True,
)
class BoxShape:
    """
    Width and height of a rectangle, in cm: a whole figure, or the share it gives one
    panel.

    Named rather than a bare pair, so which of the two is the height never has to be
    remembered; `as_mpl_shape` produces the pair for handing straight to Matplotlib.
    """

    width_cm: float
    height_cm: float

    def __post_init__(self) -> None:
        for param_name in (
            "width_cm",
            "height_cm",
        ):
            param_value = getattr(self, param_name)
            if not (param_value > 0):
                raise ValueError(f"`{param_name}` must be positive, but got {param_value}.")

    @property
    def as_mpl_shape(self) -> tuple[float, float]:
        """The pair, width first and in inches, as Matplotlib's `figsize` reads it."""
        return (
            self.width_cm / style_figure.CM_PER_INCH,
            self.height_cm / style_figure.CM_PER_INCH,
        )

    @property
    def aspect_ratio(self) -> float:
        """Width over height."""
        return self.width_cm / self.height_cm


##
## === INTERNAL HELPERS
##

DEFAULT_PANEL_SHAPE: BoxShape = BoxShape(
    width_cm=15.0,
    height_cm=10.0,
)


def _set_figure_margins(
    *,
    figure: mpl_Figure,
    figure_shape: BoxShape,
    figure_margins: style_figure.FigureMargins,
) -> None:
    """
    Leave `figure_margins` clear around the panels in `figure`.

    Margins are in pt, while Matplotlib places panels as fractions of the figure, so
    `figure_shape` is what converts between the two.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    if (figure_margins.left + figure_margins.right) >= figure_width_pt:
        raise ValueError(
            f"margins `left` + `right` ({figure_margins.left + figure_margins.right} pt)"
            f" leave no room for the panels in a figure {figure_width_pt:.1f} pt wide.",
        )
    if (figure_margins.bottom + figure_margins.top) >= figure_height_pt:
        raise ValueError(
            f"margins `bottom` + `top` ({figure_margins.bottom + figure_margins.top} pt)"
            f" leave no room for the panels in a figure {figure_height_pt:.1f} pt tall.",
        )
    ## Matplotlib positions panels from the figure's left and bottom edges, so the right
    ## and top margins are measured back from the far edge
    left_position = figure_margins.left / figure_width_pt
    right_position = 1.0 - (figure_margins.right / figure_width_pt)
    bottom_position = figure_margins.bottom / figure_height_pt
    top_position = 1.0 - (figure_margins.top / figure_height_pt)
    figure.subplots_adjust(
        left=left_position,
        right=right_position,
        bottom=bottom_position,
        top=top_position,
    )


def _compute_mpl_panel_gap(
    *,
    gap_length_pt: float,
    figure_length_pt: float,
    start_margin_pt: float,
    end_margin_pt: float,
    num_panels: int,
    param_name: str,
) -> float:
    """
    Convert the gap between two panels (pt) into the fraction of a panel Matplotlib wants.

    Matplotlib measures a gap against the panel beside it, and the panels share whatever
    the margins leave, so the gaps come out of that total length before the panels do.
    """
    total_length_pt = figure_length_pt - start_margin_pt - end_margin_pt
    total_gap_length_pt = (num_panels - 1) * gap_length_pt
    if total_gap_length_pt >= total_length_pt:
        raise ValueError(
            f"`{param_name}` ({gap_length_pt} pt) leaves no room for {num_panels} panels"
            f" in the {total_length_pt:.1f} pt the margins leave.",
        )
    panel_length_pt = (total_length_pt - total_gap_length_pt) / num_panels
    return gap_length_pt / panel_length_pt


def _set_panel_gaps(
    *,
    figure: mpl_Figure,
    figure_shape: BoxShape,
    figure_margins: style_figure.FigureMargins,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_column_gap: float,
    panel_row_gap: float,
) -> None:
    """
    Leave `panel_column_gap` and `panel_row_gap` (pt) between each pair of panels in `figure`.

    Gaps are in pt like the margins, since a gap holds the neighbouring panel's tick and
    axis labels; `figure_shape` and `figure_margins` give the length they are measured in.
    """
    figure_width_pt = style_figure.PT_PER_CM * figure_shape.width_cm
    figure_height_pt = style_figure.PT_PER_CM * figure_shape.height_cm
    figure.subplots_adjust(
        wspace=_compute_mpl_panel_gap(
            gap_length_pt=panel_column_gap,
            figure_length_pt=figure_width_pt,
            start_margin_pt=figure_margins.left,
            end_margin_pt=figure_margins.right,
            num_panels=num_panel_columns,
            param_name="panel_column_gap",
        ),
        hspace=_compute_mpl_panel_gap(
            gap_length_pt=panel_row_gap,
            figure_length_pt=figure_height_pt,
            start_margin_pt=figure_margins.bottom,
            end_margin_pt=figure_margins.top,
            num_panels=num_panel_rows,
            param_name="panel_row_gap",
        ),
    )


def _compute_panel_shape(
    *,
    figure_width_cm: float,
    num_panel_columns: int,
    panel_aspect_ratio: float,
) -> BoxShape:
    """
    Compute the share of a figure (cm) given to one panel, splitting its width by column.

    `panel_aspect_ratio` is the width / height of that share; a panel is drawn smaller than
    its share, by whatever the margins hold.
    """
    if num_panel_columns < 1:
        raise ValueError(f"`num_panel_columns` must be >= 1, but got {num_panel_columns}.")
    if not (panel_aspect_ratio > 0):
        raise ValueError(f"`panel_aspect_ratio` must be positive, but got {panel_aspect_ratio}.")
    panel_width_cm = figure_width_cm / num_panel_columns
    return BoxShape(
        height_cm=panel_width_cm / panel_aspect_ratio,
        width_cm=panel_width_cm,
    )


def _ensure_figure_sizing(
    *,
    panel_shape: BoxShape | None,
    figure_layout: style_figure.FigureLayout | None,
    panel_aspect_ratio: float | None,
) -> None:
    """
    Check that a figure is sized from a page layout, or in its own terms, but not both.

    A layout pins the figure to a share of the page, so adding panels makes each one
    smaller. `panel_shape` instead gives each panel a fixed size, so the figure grows as
    panels are added, and the figure is no longer tied to a page.
    """
    if panel_shape is None:
        return
    if figure_layout is not None:
        raise ValueError(
            "`figure_layout` and `panel_shape` are mutually exclusive: a layout sizes the"
            " figure to a share of the page, while `panel_shape` sizes each panel outright.",
        )
    if panel_aspect_ratio is not None:
        raise ValueError(
            "`panel_aspect_ratio` only applies when a figure is sized to a page;"
            " with `panel_shape` the shape of each panel is already set by it.",
        )


def _resolve_figure_layout(
    *,
    figure_layout: style_figure.FigureLayout | None,
) -> style_figure.FigureLayout:
    """The layout given, or the one set by the most recent `set_theme` call."""
    if figure_layout is None:
        return style_figure.get_figure_params().figure_layout
    return figure_layout


def _compute_figure_shape(
    *,
    panel_shape: BoxShape | None,
    figure_layout: style_figure.FigureLayout,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_aspect_ratio: float | None,
) -> BoxShape:
    """
    Size a figure (cm), either from its share of the page or from the size each panel is given.

    `panel_shape` is what chooses between the two; `_ensure_figure_sizing` is what checks
    the two ways were not both asked for.
    """
    if (num_panel_rows < 1) or (num_panel_columns < 1):
        raise ValueError("`num_panel_rows` and `num_panel_columns` must both be >= 1.")
    if panel_shape is None:
        if panel_aspect_ratio is None:
            panel_aspect_ratio = DEFAULT_PANEL_SHAPE.aspect_ratio
        figure_panel_shape = _compute_panel_shape(
            figure_width_cm=figure_layout.figure_width.width_cm,
            num_panel_columns=num_panel_columns,
            panel_aspect_ratio=panel_aspect_ratio,
        )
    else:
        figure_panel_shape = panel_shape
    return BoxShape(
        width_cm=figure_panel_shape.width_cm * num_panel_columns,
        height_cm=figure_panel_shape.height_cm * num_panel_rows,
    )


##
## === FIGURE FACTORY
##


@overload
def create_figure(
    *,
    num_panel_rows: None = None,
    num_panel_columns: None = None,
    panel_shape: BoxShape | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
) -> tuple[mpl_Figure, Panel]:
    ...


@overload
def create_figure(
    *,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_shape: BoxShape | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_column_gap: float = 10.0,
    panel_row_gap: float = 10.0,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_Figure, PanelGrid]:
    ...


def create_figure(
    *,
    num_panel_rows: int | None = None,
    num_panel_columns: int | None = None,
    panel_shape: BoxShape | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_column_gap: float = 10.0,
    panel_row_gap: float = 10.0,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_Figure, Panel | PanelGrid]:
    """
    Create a Matplotlib figure and Axis / Axes grid.

    Overloads:
        - create_figure() -> (figure, panel)
        - create_figure(num_panel_rows=N, num_panel_columns=M) -> (figure, panel_grid) of shape (N, M)

    Sizing
    ------
    A figure is sized in one of two ways, and asking for both is refused.

    By default it takes its share of the page, from `figure_layout` or from the one
    `set_theme` last set. The figure width is then fixed, so adding columns makes each
    panel narrower, and `panel_aspect_ratio` sets the shape of the share each panel gets.

    Passing `panel_shape` instead sizes each panel outright in cm, so the figure grows as
    panels are added and is no longer tied to a page.

    Either way, `panel_shape` is the share of the figure a panel gets, not the size it is
    drawn at: tick labels and axis labels are held in margins taken out of that share, so
    the panel itself is drawn smaller. A colorbar is placed beyond the panel rather than
    within those margins, and so can reach past the figure edge.

    Notes
    -----
    - If `num_panel_rows` and `num_panel_columns` are both None (or omitted), a single-panel
      figure is created and a single Panel is returned.
    - If both are given as integers, a grid is created and a 2D object-dtype `PanelGrid` is
      returned; 1x1 is refused, since that is the single-panel case.
    - Mixed None/int specifications are not allowed.
    """
    if (num_panel_rows is None) and (num_panel_columns is None):
        num_panel_rows = 1
        num_panel_columns = 1
    elif (num_panel_rows is None) or (num_panel_columns is None):
        raise ValueError(
            "Either specify both `num_panel_rows` and `num_panel_columns`, or neither."
            " Mixed None/int combinations are not supported.",
        )
    else:
        validate_types.ensure_finite_int(
            param=num_panel_rows,
            param_name="num_panel_rows",
            require_positive=True,
        )
        validate_types.ensure_finite_int(
            param=num_panel_columns,
            param_name="num_panel_columns",
            require_positive=True,
        )
        if (num_panel_rows == 1) and (num_panel_columns == 1):
            raise ValueError(
                "For a single-panel figure, omit `num_panel_rows` and `num_panel_columns` so that"
                " a single Panel is returned instead of a 1x1 panel grid.",
            )
    ## a 1x1 grid is rejected above, so both being 1 means the arguments were omitted
    is_single_panel = (num_panel_rows == 1) and (num_panel_columns == 1)
    _ensure_figure_sizing(
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        panel_aspect_ratio=panel_aspect_ratio,
    )
    figure_layout = _resolve_figure_layout(figure_layout=figure_layout)
    figure_shape = _compute_figure_shape(
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_aspect_ratio=panel_aspect_ratio,
    )
    figure, panels = mpl_plot.subplots(
        nrows=num_panel_rows,
        ncols=num_panel_columns,
        figsize=figure_shape.as_mpl_shape,
        sharex=share_x_axis,
        sharey=share_y_axis,
        ## squeeze a 1x1 grid down to the single Panel the caller asked for
        squeeze=is_single_panel,
    )
    _set_figure_margins(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_layout.figure_margins,
    )
    if is_single_panel:
        return figure, panels
    _set_panel_gaps(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=figure_layout.figure_margins,
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_column_gap=panel_column_gap,
        panel_row_gap=panel_row_gap,
    )
    panel_grid: PanelGrid = numpy.asarray(panels, dtype=object)
    return figure, panel_grid


def create_figure_grid(
    *,
    num_panel_rows: int = 1,
    num_panel_columns: int = 1,
    panel_shape: BoxShape | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_column_gap: float = 10.0,
    panel_row_gap: float = 10.0,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_Figure, PanelGrid]:
    """
    Like `create_figure`, but always returns a 2D panel grid of shape (num_panel_rows, num_panel_columns), so
    callers can always index panels as panel_grid[row, col].
    """
    if (num_panel_rows == 1) and (num_panel_columns == 1):
        figure, panel = create_figure(
            panel_shape=panel_shape,
            figure_layout=figure_layout,
            panel_aspect_ratio=panel_aspect_ratio,
        )
        panel_grid: PanelGrid = numpy.asarray([[panel]], dtype=object)
        return figure, panel_grid
    figure, panel_grid = create_figure(
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        panel_aspect_ratio=panel_aspect_ratio,
        panel_column_gap=panel_column_gap,
        panel_row_gap=panel_row_gap,
        share_x_axis=share_x_axis,
        share_y_axis=share_y_axis,
    )
    return figure, panel_grid


##
## === AXIS HELPERS
##

_Side = box_positions.Positions.Side


@dataclass(frozen=True)
class PanelBounds:
    """Bounding box for a panel in figure coordinates."""

    x_min: float
    y_min: float
    x_width: float
    y_width: float


def compute_neighbouring_panel_bounds(
    *,
    panel: Panel,
    side: _Side = box_positions.Positions.Side.Right,
    gap: float = 0.1,
    thickness: float = 1.0,
    length: float = 1.0,
) -> PanelBounds:
    """
    Compute figure bounds for a panel placed neighbouring `panel`.

    The new panel sits on the `side` of `panel`, offset by `gap` (in figure coordinates).
    `thickness` sets its extent perpendicular to `side`, as a fraction of `panel`'s
    corresponding dimension. `length` sets its extent parallel to `side`, also as a
    fraction, centered on `panel`'s edge.
    """
    box = panel.get_position()
    if side in (_Side.Left, _Side.Right):
        x_width = box.width * thickness
        y_width = box.height * length
        if side == _Side.Right:
            return PanelBounds(
                x_min=box.x1 + gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return PanelBounds(
                x_min=box.x0 - x_width - gap,
                y_min=box.y0 + (box.height - y_width) / 2,
                x_width=x_width,
                y_width=y_width,
            )
    elif side in (_Side.Top, _Side.Bottom):
        x_width = box.width * length
        y_width = box.height * thickness
        if side == _Side.Top:
            return PanelBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y1 + gap,
                x_width=x_width,
                y_width=y_width,
            )
        else:
            return PanelBounds(
                x_min=box.x0 + (box.width - x_width) / 2,
                y_min=box.y0 - y_width - gap,
                x_width=x_width,
                y_width=y_width,
            )
    else:
        raise ValueError(f"unexpected side: {side!r}.")  # pyright: ignore[reportUnreachable]


def add_inset_panel(
    *,
    panel: Panel,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    text_size: float | None = None,
    x_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Top,
    y_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
) -> Panel:
    """Add an inset Axis to `panel`."""
    x_label_side = validate_box_positions.as_box_side(x_label_alignment)
    y_label_side = validate_box_positions.as_box_side(y_label_alignment)
    inset_panel = panel.inset_axes(bounds)
    if text_size is None:
        text_size = rcParams["axes.labelsize"]
    if x_label is not None:
        inset_panel.set_xlabel(
            xlabel=x_label,
            fontsize=text_size,
        )
        inset_panel.xaxis.set_label_position(x_label_side.value)  # pyright: ignore[reportArgumentType]
    if y_label is not None:
        inset_panel.set_ylabel(
            ylabel=y_label,
            fontsize=text_size,
        )
        inset_panel.yaxis.set_label_position(y_label_side.value)  # pyright: ignore[reportArgumentType]
    inset_panel.tick_params(
        axis="x",
        labeltop=(x_label_side is box_positions.Positions.Side.Top),
        labelbottom=(x_label_side is box_positions.Positions.Side.Bottom),
        top=True,
        bottom=True,
    )
    if x_label_side is box_positions.Positions.Side.Top:
        inset_panel.xaxis.tick_top()
    inset_panel.tick_params(
        axis="y",
        labelleft=(y_label_side is box_positions.Positions.Side.Left),
        labelright=(y_label_side is box_positions.Positions.Side.Right),
        left=True,
        right=True,
    )
    if y_label_side is box_positions.Positions.Side.Right:
        inset_panel.yaxis.tick_right()
    return inset_panel


##
## === IO HELPERS
##


def save_figure(
    *,
    figure: mpl_Figure,
    figure_path: str | Path,
    pixels_per_cm: float = style_figure.DEFAULT_PIXELS_PER_CM,
    verbose: bool = True,
) -> None:
    """
    Save `figure` to `figure_path`; close it.

    `pixels_per_cm` is how finely a raster is sampled, in the cm a figure is sized in;
    Matplotlib wants it per inch. Accepts `.png` or `.pdf` paths; errors are logged
    rather than raised.
    """
    if not str(figure_path).endswith(".png") and not str(figure_path).endswith(".pdf"):
        raise ValueError("figures must end with `.png` or `.pdf`.")
    if not (pixels_per_cm > 0):
        raise ValueError(f"`pixels_per_cm` must be positive, but got {pixels_per_cm}.")
    try:
        pixels_per_inch = style_figure.CM_PER_INCH * pixels_per_cm
        figure.savefig(figure_path, dpi=pixels_per_inch)
        if verbose:
            manage_log.log_action(
                title="Save figure",
                outcome=manage_log.ActionOutcome.SUCCESS,
                message="Saved figure.",
                notes={"file": str(figure_path)},
            )
    except FileNotFoundError as exception:
        manage_log.log_error(text=f"FileNotFoundError: {exception}")
    except PermissionError as exception:
        manage_log.log_error(
            text=f"PermissionError: You do not have permission to save to: {figure_path}",
            notes={"details": str(exception)},
        )
    except IOError as exception:
        manage_log.log_error(
            text=f"IOError: An error occurred while trying to save the figure to: {figure_path}",
            notes={"details": str(exception)},
        )
    except Exception as exception:
        manage_log.log_error(text=f"Unexpected error while saving the figure to {figure_path}: {exception}")
    finally:
        mpl_plot.close(figure)


def animate_frames_to_video(
    *,
    frames_dir: str | Path,
    video_path: str | Path,
    pattern: str = "frame_*.png",
    frames_per_second: int = 30,
    timeout_seconds: int = 60,
) -> None:
    """
    Combine the frames in `frames_dir` matching `pattern` into a video at `video_path`.

    Requires `ffmpeg` on the system path. Creates the parent directory of `video_path` if
    needed.
    """
    frames_dir = Path(frames_dir)
    video_path = Path(video_path)
    manage_io.create_directory(
        directory=video_path.parent,
        verbose=False,
    )
    args = " ".join(
        [
            "-hide_banner",  # less stdout
            "-loglevel error",  # only errors
            "-y",  # overwrite output
            f"-framerate {frames_per_second}",  # input rate (put before -i)
            "-pattern_type glob",  # enable glob input (put before -i)
            f'-i "{pattern}"',  # input pattern (e.g., frame_*.png)
            '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"',  # enforce even dims
            "-c:v mpeg4 -q:v 3",  # codec + quality
            "-pix_fmt yuv420p",  # broad compatibility
            f"-r {frames_per_second}",  # output rate
        ],
    )
    cmd = f'ffmpeg {args} "{video_path}"'
    manage_shell.execute_shell_command(
        command=cmd,
        working_directory=frames_dir,
        timeout_seconds=timeout_seconds,
    )
    manage_log.log_action(
        title="Save animation",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Saved video.",
        notes={"file": str(video_path)},
    )


## } MODULE
