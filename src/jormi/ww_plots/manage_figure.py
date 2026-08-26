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
import dataclasses

from pathlib import Path
from typing import overload

## third-party
import numpy
from matplotlib import pyplot as mpl_plot
from matplotlib.figure import Figure as mpl_Figure

## local
from jormi.ww_io import (
    manage_io,
    manage_log,
    manage_shell,
)
from jormi.ww_plots import _layout_figure, style_figure
from jormi.ww_validation import validate_box_positions, validate_types
from jormi.ww_types import box_positions

## how a figure is laid out lives in `_layout_figure`, which is jormi's own and not meant to
## be imported directly; what a caller needs of it is named here instead
from jormi.ww_plots._layout_figure import (
    BoxShape,
    Panel,
    PanelBounds,
    PanelGrid,
    as_panel_list,
    compute_colorbar_thickness_share,
    compute_neighbouring_panel_bounds,
    fit_figure_to_content,
    register_colorbar,
    register_shared_label,
)


##
## === INTERNAL HELPERS
##

DEFAULT_PANEL_SHAPE: BoxShape = BoxShape(
    width_cm=15.0,
    height_cm=10.0,
)



def _ensure_figure_sizing(
    *,
    panel_shape: BoxShape | None,
    figure_layout: style_figure.FigureLayout | None,
    panel_aspect: float | None,
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
    if panel_aspect is not None:
        raise ValueError(
            "`panel_aspect` only applies when a figure is sized to a page;"
            " with `panel_shape` the shape of each panel is already set by it.",
        )


def _compose_figure_params(
    *,
    figure_params: style_figure.FigureParams | None,
    figure_layout: style_figure.FigureLayout | None,
    panel_column_gap: float | None,
    panel_row_gap: float | None,
) -> style_figure.FigureParams:
    """
    Fold the arguments a figure is built from into one set of params describing it.

    Each argument overrides the matching part of `figure_params`, which in turn stands in
    for the active style when it is not given. Composing them into one object is what lets
    a colorbar added later sit at the same gap the panels were spaced by.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if figure_layout is None:
        figure_layout = figure_params.figure_layout
    panel_gaps = figure_layout.panel_gaps
    if (panel_column_gap is not None) or (panel_row_gap is not None):
        panel_gaps = style_figure.PanelGaps(
            column=(panel_gaps.column if panel_column_gap is None else panel_column_gap),
            row=(panel_gaps.row if panel_row_gap is None else panel_row_gap),
        )
    return dataclasses.replace(
        figure_params,
        figure_layout=dataclasses.replace(
            figure_layout,
            panel_gaps=panel_gaps,
        ),
    )


def _compute_figure_shape(
    *,
    panel_shape: BoxShape | None,
    figure_layout: style_figure.FigureLayout,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_aspect: float | None,
) -> BoxShape:
    """
    Size a figure (cm), either from its share of the page or from the size each panel is given.

    `panel_shape` is what chooses between the two; `_ensure_figure_sizing` is what checks
    the two ways were not both asked for.

    Sized to a page, this is only where the figure starts: `fit_figure_to_content` measures
    what it holds and sizes it again, so that `panel_aspect` is what the panel is drawn at.
    """
    if (num_panel_rows < 1) or (num_panel_columns < 1):
        raise ValueError("`num_panel_rows` and `num_panel_columns` must both be >= 1.")
    if panel_shape is None:
        panel_width_cm = figure_layout.figure_width.width_cm / num_panel_columns
        figure_panel_shape = BoxShape(
            width_cm=panel_width_cm,
            height_cm=panel_width_cm / (panel_aspect or DEFAULT_PANEL_SHAPE.aspect_ratio),
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
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect: float | None = None,
) -> tuple[mpl_Figure, Panel]:
    ...


@overload
def create_figure(
    *,
    num_panel_rows: int,
    num_panel_columns: int,
    panel_shape: BoxShape | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect: float | None = None,
    panel_column_gap: float | None = None,
    panel_row_gap: float | None = None,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_Figure, PanelGrid]:
    ...


def create_figure(
    *,
    num_panel_rows: int | None = None,
    num_panel_columns: int | None = None,
    panel_shape: BoxShape | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect: float | None = None,
    panel_column_gap: float | None = None,
    panel_row_gap: float | None = None,
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
    `set_figure_params` last set. The figure width is then fixed, so adding columns makes each
    panel narrower, and `panel_aspect` sets the shape each panel is drawn at.

    Passing `panel_shape` instead sizes each panel outright in cm, so the figure grows as
    panels are added and is no longer tied to a page.

    Sized to a page, the figure is fitted to what it holds when it is saved: its labels and
    any colorbar are measured, and the margins and the figure height follow from them, so
    `panel_aspect` describes the panel itself rather than a share it is drawn inside.

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
    if (panel_aspect is not None) and not (panel_aspect > 0):
        raise ValueError(f"`panel_aspect` must be positive, but got {panel_aspect}.")
    _ensure_figure_sizing(
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        panel_aspect=panel_aspect,
    )
    ## everything the figure is built from, gathered into one object and made active, so
    ## that whatever is drawn onto the figure afterwards is styled and spaced to match it
    figure_params = _compose_figure_params(
        figure_params=figure_params,
        figure_layout=figure_layout,
        panel_column_gap=panel_column_gap,
        panel_row_gap=panel_row_gap,
    )
    style_figure.set_figure_params(figure_params=figure_params)
    figure_layout = figure_params.figure_layout
    figure_shape = _compute_figure_shape(
        panel_shape=panel_shape,
        figure_layout=figure_layout,
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_aspect=panel_aspect,
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
    _layout_figure._set_figure_margins(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=_layout_figure.DEFAULT_FIGURE_MARGINS,
    )
    ## the gap arguments were folded into the composed layout, so read them back from it
    panel_gaps = figure_layout.panel_gaps
    if panel_aspect is not None:
        ## the shape set above is provisional: `save_figure` measures the labels and solves
        ## for the panel width and the figure height that give `panel_aspect` as drawn
        _layout_figure._FIGURE_FITS[figure] = _layout_figure.FigureFit(
            num_panel_rows=num_panel_rows,
            num_panel_columns=num_panel_columns,
            panel_aspect=panel_aspect,
            figure_padding=figure_layout.figure_padding,
            panel_column_gap=panel_gaps.column,
            panel_row_gap=panel_gaps.row,
        )
    if is_single_panel:
        return figure, panels
    _layout_figure._set_panel_gaps(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=_layout_figure.DEFAULT_FIGURE_MARGINS,
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_column_gap=panel_gaps.column,
        panel_row_gap=panel_gaps.row,
    )
    panel_grid: PanelGrid = numpy.asarray(panels, dtype=object)
    return figure, panel_grid


def create_figure_grid(
    *,
    num_panel_rows: int = 1,
    num_panel_columns: int = 1,
    panel_shape: BoxShape | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect: float | None = None,
    panel_column_gap: float | None = None,
    panel_row_gap: float | None = None,
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
            figure_params=figure_params,
            figure_layout=figure_layout,
                panel_aspect=panel_aspect,
        )
        panel_grid: PanelGrid = numpy.asarray([[panel]], dtype=object)
        return figure, panel_grid
    figure, panel_grid = create_figure(
        num_panel_rows=num_panel_rows,
        num_panel_columns=num_panel_columns,
        panel_shape=panel_shape,
        figure_params=figure_params,
        figure_layout=figure_layout,
        panel_aspect=panel_aspect,
        panel_column_gap=panel_column_gap,
        panel_row_gap=panel_row_gap,
        share_x_axis=share_x_axis,
        share_y_axis=share_y_axis,
    )
    return figure, panel_grid


##
## === AXIS HELPERS

def add_inset_panel(
    *,
    panel: Panel,
    bounds: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    text_size: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    x_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Top,
    y_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
) -> Panel:
    """Add an inset Axis to `panel`; `text_size` defaults to the active axis-label size."""
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    x_label_side = validate_box_positions.as_box_side(x_label_alignment)
    y_label_side = validate_box_positions.as_box_side(y_label_alignment)
    inset_panel = panel.inset_axes(bounds)
    if text_size is None:
        text_size = figure_params.text_size_params.axis_label_size
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
    pixels_per_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    verbose: bool = True,
) -> None:
    """
    Save `figure` to `figure_path`; close it.

    `pixels_per_cm` is how finely a raster is sampled, in the cm a figure is sized in;
    Matplotlib wants it per inch. It defaults to the active style's, so a style that
    asks for a density gets it. Accepts `.png` or `.pdf` paths; errors are logged
    rather than raised.
    """
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if pixels_per_cm is None:
        pixels_per_cm = figure_params.save_params.pixels_per_cm
    if not str(figure_path).endswith(".png") and not str(figure_path).endswith(".pdf"):
        raise ValueError("figures must end with `.png` or `.pdf`.")
    if not (pixels_per_cm > 0):
        raise ValueError(f"`pixels_per_cm` must be positive, but got {pixels_per_cm}.")
    try:
        ## a figure built with `panel_aspect` is sized here rather than at creation, since
        ## only now does every label it has to hold exist to be measured
        fit_figure_to_content(figure=figure)
        pixels_per_inch = style_figure.CM_PER_INCH * pixels_per_cm
        if figure_params.save_params.transparent_background:
            figure.savefig(figure_path, dpi=pixels_per_inch)
        else:
            ## take the colours off the figure rather than from the active style, so a
            ## theme set after this figure was built cannot repaint it on the way out
            figure.savefig(
                figure_path,
                dpi=pixels_per_inch,
                facecolor=figure.get_facecolor(),
                edgecolor=figure.get_edgecolor(),
            )
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
