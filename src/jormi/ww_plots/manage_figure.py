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
import pathlib
import typing

## third-party
import numpy

from matplotlib import figure as mpl_figure
from matplotlib import pyplot as mpl_plot

## local
from jormi.ww_io import (
    manage_io,
    manage_log,
    manage_shell,
)
from jormi.ww_plots import _layout_figure, style_figure
from jormi.ww_validation import validate_box_positions, validate_types
from jormi.ww_types import box_positions

##
## === RE-EXPORTS
##

## how a figure is laid out lives in `_layout_figure`, which is internal to jormi and not
## meant to be imported directly; what a caller needs of it is named here instead
FigureMargins = _layout_figure.FigureMargins
FigureSize = _layout_figure.FigureSize
Panel = _layout_figure.Panel
PanelBounds = _layout_figure.PanelBounds
PanelGrid = _layout_figure.PanelGrid
get_figure = _layout_figure.get_figure
as_panel_list = _layout_figure.as_panel_list
compute_colorbar_thickness_fraction = _layout_figure.compute_colorbar_thickness_fraction
compute_panel_fractional_bounds = _layout_figure.compute_panel_fractional_bounds
register_colorbar = _layout_figure.register_colorbar
register_shared_label = _layout_figure.register_shared_label

##
## === INTERNAL HELPERS
##

## the shape a panel takes when a caller does not say; 3:2 reads well for a lone plot
DEFAULT_PANEL_ASPECT_RATIO: float = 1.5


def _ensure_figure_sizing(
    *,
    panel_width_cm: float | None,
    figure_layout: style_figure.FigureLayout | None,
    panel_aspect_ratio: float | None,
) -> None:
    """Check that only one parameter sets the length scale."""
    if panel_width_cm is None:
        return
    if figure_layout is not None:
        raise ValueError(
            "`figure_layout` and `panel_width_cm` are mutually exclusive: a layout sizes the"
            " figure to a share of the page, while `panel_width_cm` sizes each panel outright.",
        )
    if not (panel_width_cm > 0):
        raise ValueError(f"`panel_width_cm` must be positive, but got {panel_width_cm}.")


def _compose_figure_params(
    *,
    figure_params: style_figure.FigureParams | None,
    figure_layout: style_figure.FigureLayout | None,
    panel_row_gap_pt: float | None,
    panel_col_gap_pt: float | None,
) -> style_figure.FigureParams:
    """Resolve FigureParams from the currently active figure style + provided overrides."""
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    if figure_layout is None:
        figure_layout = figure_params.figure_layout
    panel_gaps = figure_layout.panel_gaps
    if (panel_row_gap_pt is not None) or (panel_col_gap_pt is not None):
        panel_gaps = style_figure.PanelGaps(
            row_pt=(panel_gaps.row_pt if panel_row_gap_pt is None else panel_row_gap_pt),
            col_pt=(panel_gaps.col_pt if panel_col_gap_pt is None else panel_col_gap_pt),
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
    panel_width_cm: float | None,
    figure_layout: style_figure.FigureLayout,
    num_panel_rows: int,
    num_panel_cols: int,
    panel_aspect_ratio: float | None,
) -> _layout_figure.BoxShape:
    """Size a figure (in cm) from whichever length was pinned."""
    if (num_panel_rows < 1) or (num_panel_cols < 1):
        raise ValueError("`num_panel_rows` and `num_panel_cols` must both be >= 1.")
    if panel_width_cm is None:
        panel_width_cm = figure_layout.figure_width.width_cm / num_panel_cols
    panel_height_cm = panel_width_cm / (panel_aspect_ratio or DEFAULT_PANEL_ASPECT_RATIO)
    return _layout_figure.BoxShape(
        width_cm=panel_width_cm * num_panel_cols,
        height_cm=panel_height_cm * num_panel_rows,
    )


##
## === FIGURE FACTORY
##


@typing.overload
def create_figure(
    *,
    num_panel_rows: None = None,
    num_panel_cols: None = None,
    panel_width_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
) -> tuple[mpl_figure.Figure, Panel]:
    ...


@typing.overload
def create_figure(
    *,
    num_panel_rows: int,
    num_panel_cols: int,
    panel_width_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_row_gap_pt: float | None = None,
    panel_col_gap_pt: float | None = None,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_figure.Figure, PanelGrid]:
    ...


def create_figure(
    *,
    num_panel_rows: int | None = None,
    num_panel_cols: int | None = None,
    panel_width_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_row_gap_pt: float | None = None,
    panel_col_gap_pt: float | None = None,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_figure.Figure, Panel | PanelGrid]:
    """
    Create a Matplotlib figure and Axis / Axes grid.

    `num_panel_rows` and `num_panel_cols` must both be given or both omitted: omitting
    both returns a single Panel; giving both returns a 2D PanelGrid. 1x1 is refused, since
    that is the single-panel case.
    """
    if (num_panel_rows is None) and (num_panel_cols is None):
        num_panel_rows = 1
        num_panel_cols = 1
    elif (num_panel_rows is None) or (num_panel_cols is None):
        raise ValueError(
            "Either specify both `num_panel_rows` and `num_panel_cols`, or neither."
            " Mixed None/int combinations are not supported.",
        )
    else:
        validate_types.ensure_finite_int(
            param=num_panel_rows,
            param_name="num_panel_rows",
            require_positive=True,
        )
        validate_types.ensure_finite_int(
            param=num_panel_cols,
            param_name="num_panel_cols",
            require_positive=True,
        )
        if (num_panel_rows == 1) and (num_panel_cols == 1):
            raise ValueError(
                "For a single-panel figure, omit `num_panel_rows` and `num_panel_cols` so that"
                " a single Panel is returned instead of a 1x1 panel grid.",
            )
    ## a 1x1 grid is rejected above, so both being 1 means the arguments were omitted
    is_single_panel = (num_panel_rows == 1) and (num_panel_cols == 1)
    if (panel_aspect_ratio is not None) and not (panel_aspect_ratio > 0):
        raise ValueError(f"`panel_aspect_ratio` must be positive, but got {panel_aspect_ratio}.")
    _ensure_figure_sizing(
        panel_width_cm=panel_width_cm,
        figure_layout=figure_layout,
        panel_aspect_ratio=panel_aspect_ratio,
    )
    ## everything the figure is built from, gathered into one object and made active, so
    ## that whatever is drawn onto the figure afterwards is styled and spaced to match it
    figure_params = _compose_figure_params(
        figure_params=figure_params,
        figure_layout=figure_layout,
        panel_row_gap_pt=panel_row_gap_pt,
        panel_col_gap_pt=panel_col_gap_pt,
    )
    style_figure.set_figure_params(figure_params=figure_params)
    figure_layout = figure_params.figure_layout
    figure_shape = _compute_figure_shape(
        panel_width_cm=panel_width_cm,
        figure_layout=figure_layout,
        num_panel_rows=num_panel_rows,
        num_panel_cols=num_panel_cols,
        panel_aspect_ratio=panel_aspect_ratio,
    )
    figure, panels = mpl_plot.subplots(
        nrows=num_panel_rows,
        ncols=num_panel_cols,
        figsize=figure_shape.as_mpl_shape,
        sharex=share_x_axis,
        sharey=share_y_axis,
        ## squeeze a 1x1 grid down to the single Panel the caller asked for
        squeeze=is_single_panel,
    )
    _layout_figure.set_figure_margins(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=_layout_figure.INITIAL_FIGURE_MARGINS,
    )
    ## the gap arguments were folded into the composed layout, so read them back from it
    panel_gaps = figure_layout.panel_gaps
    ## the shape set above is provisional; `save_figure` measures the labels and resizes it
    panel_width_pt = None if (panel_width_cm is None) else style_figure.PT_PER_CM * panel_width_cm
    _layout_figure.RESOLVED_LAYOUTS[figure] = _layout_figure.ResolvedLayout(
        num_panel_rows=num_panel_rows,
        num_panel_cols=num_panel_cols,
        panel_width_pt=panel_width_pt,
        panel_aspect_ratio=panel_aspect_ratio or DEFAULT_PANEL_ASPECT_RATIO,
        figure_padding=figure_layout.figure_padding,
        panel_row_gap_pt=panel_gaps.row_pt,
        panel_col_gap_pt=panel_gaps.col_pt,
    )
    if is_single_panel:
        return figure, panels
    _layout_figure.set_panel_gaps(
        figure=figure,
        figure_shape=figure_shape,
        figure_margins=_layout_figure.INITIAL_FIGURE_MARGINS,
        num_panel_rows=num_panel_rows,
        num_panel_cols=num_panel_cols,
        panel_row_gap_pt=panel_gaps.row_pt,
        panel_col_gap_pt=panel_gaps.col_pt,
    )
    panel_grid: PanelGrid = numpy.asarray(panels, dtype=object)
    return figure, panel_grid


def create_figure_grid(
    *,
    num_panel_rows: int = 1,
    num_panel_cols: int = 1,
    panel_width_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    figure_layout: style_figure.FigureLayout | None = None,
    panel_aspect_ratio: float | None = None,
    panel_row_gap_pt: float | None = None,
    panel_col_gap_pt: float | None = None,
    share_x_axis: bool = False,
    share_y_axis: bool = False,
) -> tuple[mpl_figure.Figure, PanelGrid]:
    """
    Like `create_figure`, but always returns a 2D panel grid of shape (num_panel_rows, num_panel_cols), so
    callers can always index panels as panel_grid[row, col].
    """
    if (num_panel_rows == 1) and (num_panel_cols == 1):
        figure, panel = create_figure(
            panel_width_cm=panel_width_cm,
            figure_params=figure_params,
            figure_layout=figure_layout,
            panel_aspect_ratio=panel_aspect_ratio,
        )
        panel_grid: PanelGrid = numpy.asarray([[panel]], dtype=object)
        return figure, panel_grid
    figure, panel_grid = create_figure(
        num_panel_rows=num_panel_rows,
        num_panel_cols=num_panel_cols,
        panel_width_cm=panel_width_cm,
        figure_params=figure_params,
        figure_layout=figure_layout,
        panel_aspect_ratio=panel_aspect_ratio,
        panel_row_gap_pt=panel_row_gap_pt,
        panel_col_gap_pt=panel_col_gap_pt,
        share_x_axis=share_x_axis,
        share_y_axis=share_y_axis,
    )
    return figure, panel_grid


##
## === AXIS HELPERS


def add_inset_panel(
    *,
    panel: Panel,
    bounds_fraction: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.5),
    x_label: str | None = None,
    y_label: str | None = None,
    text_size_pt: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    x_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Top,
    y_label_alignment: box_positions.Positions.PositionLike = box_positions.Positions.Side.Right,
) -> Panel:
    """
    Add an inset Axis to `panel`; `text_size_pt` defaults to the active axis-label size.

    `bounds_fraction` is (x, y, width, height) in the coordinates of the panel itself, so
    an inset sits inside its panel when they fall in [0, 1].
    """
    validate_types.ensure_tuple_of_numbers(
        param=bounds_fraction,
        param_name="bounds_fraction",
        seq_length=4,
    )
    for param_name, param_value in zip(
        (
            "bounds_fraction[2]",
            "bounds_fraction[3]",
        ),
            bounds_fraction[2:],
    ):
        validate_types.ensure_finite_float(
            param=param_value,
            param_name=param_name,
            allow_none=False,
            require_positive=True,
            allow_zero=False,
        )
    if figure_params is None:
        figure_params = style_figure.get_figure_params()
    x_label_side = validate_box_positions.as_box_side(x_label_alignment)
    y_label_side = validate_box_positions.as_box_side(y_label_alignment)
    inset_panel = panel.inset_axes(bounds_fraction)
    if text_size_pt is None:
        text_size_pt = figure_params.text_size_params.axis_label_size_pt
    if x_label is not None:
        inset_panel.set_xlabel(
            xlabel=x_label,
            fontsize=text_size_pt,
        )
        inset_panel.xaxis.set_label_position(x_label_side.value)  # pyright: ignore[reportArgumentType]
    if y_label is not None:
        inset_panel.set_ylabel(
            ylabel=y_label,
            fontsize=text_size_pt,
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
    figure: mpl_figure.Figure,
    figure_path: str | pathlib.Path,
    pixels_per_cm: float | None = None,
    figure_params: style_figure.FigureParams | None = None,
    verbose: bool = True,
) -> None:
    """
    Save `figure` to `figure_path`; close it.

    `pixels_per_cm` is how finely a raster is sampled, in the cm a figure is sized in;
    Matplotlib wants it per inch. It defaults to the value the active style asks for.
    Accepts `.png` or `.pdf` paths; errors are logged rather than raised.
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
        ## a figure built with `panel_aspect_ratio` is sized here rather than at creation, since
        ## only now does every label it has to hold exist to be measured
        _layout_figure.fit_figure_to_content(figure=figure)
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
    frames_dir: str | pathlib.Path,
    video_path: str | pathlib.Path,
    pattern: str = "frame_*.png",
    frames_per_second: int = 30,
    timeout_seconds: int = 60,
) -> None:
    """
    Combine the frames in `frames_dir` matching `pattern` into a video at `video_path`.

    Requires `ffmpeg` on the system path. Creates the parent directory of `video_path` if
    needed.
    """
    frames_dir = pathlib.Path(frames_dir)
    video_path = pathlib.Path(video_path)
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
