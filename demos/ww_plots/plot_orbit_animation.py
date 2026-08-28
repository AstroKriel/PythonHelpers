## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

from numpy.typing import NDArray

## local
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import (
    annotate_panel,
    manage_figure,
    style_figure,
)
from jormi.ww_types import box_positions

##
## === CONSTANTS
##

## one period, sampled so that the last frame lands just before the first repeats
NUM_FRAMES = 60
FRAMES_PER_SECOND = 30

ORBIT_RADIUS = 1.0
ORBIT_COLOR = "red"
## room around the orbit, so the marker never touches the panel frame
AXIS_LIMIT = 1.25

PIXELS_PER_CM = 100.0
## the frame has no page to be a fraction of, so the panel is what sets the scale
PANEL_WIDTH_CM = 6.0
PANEL_ASPECT = 1.0

##
## === DEMO DATA
##


def generate_orbit_path() -> tuple[NDArray[Any], NDArray[Any]]:
    """The circle the marker travels along, closed so it joins back on itself."""
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 256)
    return (
        ORBIT_RADIUS * numpy.cos(angles),
        ORBIT_RADIUS * numpy.sin(angles),
    )


def compute_frame_angles() -> NDArray[Any]:
    """One period of angles, dropping the endpoint so the loop does not repeat a frame."""
    return numpy.linspace(0.0, 2.0 * numpy.pi, NUM_FRAMES, endpoint=False)


##
## === HELPER FUNCTIONS
##


def draw_frame(
    *,
    frames_dir: Path,
    frame_index: int,
    angle: float,
    path_x_values: NDArray[Any],
    path_y_values: NDArray[Any],
) -> None:
    """Draw the marker at `angle` along the orbit, and save it as one frame."""
    figure, panel = manage_figure.create_figure(
        panel_aspect_ratio=PANEL_ASPECT,
        panel_width_cm=PANEL_WIDTH_CM,
    )
    annotate_panel.overlay_curve(
        panel=panel,
        x_values=path_x_values,
        y_values=path_y_values,
        linestyle="-",
    )
    panel.plot(
        ORBIT_RADIUS * numpy.cos(angle),
        ORBIT_RADIUS * numpy.sin(angle),
        marker="o",
        ls="",
        color=ORBIT_COLOR,
    )
    annotate_panel.add_text(
        panel=panel,
        x_pos_fraction=0.5,
        y_pos_fraction=0.5,
        label=rf"$\theta = {angle / numpy.pi:.2f}\,\pi$",
        x_alignment=box_positions.Positions.Center.Center,
        y_alignment=box_positions.Positions.Center.Center,
    )
    ## every frame is drawn in the same window, so the marker moves rather than the axes
    panel.set_xlim((-AXIS_LIMIT, AXIS_LIMIT))
    panel.set_ylim((-AXIS_LIMIT, AXIS_LIMIT))
    panel.set_aspect("equal")
    panel.set_xlabel(r"$x$")
    panel.set_ylabel(r"$y$")
    manage_figure.save_figure(
        figure=figure,
        figure_path=frames_dir / f"frame_{frame_index:04d}.png",
        pixels_per_cm=PIXELS_PER_CM,
        verbose=False,
    )


def clear_stale_frames(
    *,
    frames_dir: Path,
) -> None:
    """
    Remove frames left by an earlier run.

    The frames are collected by glob, so one left over from a longer run would be picked
    up as if it belonged to this one.
    """
    stale_frame_paths = manage_io.filter_directory(
        directory=frames_dir,
        prefix="frame",
        suffix=".png",
        include_folders=False,
    )
    for stale_frame_path in stale_frame_paths:
        manage_io.delete_file(
            directory=frames_dir,
            file_name=stale_frame_path.name,
            verbose=False,
        )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(mode=manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    demo_dir = Path(__file__).parent
    frames_dir = demo_dir / "frames"
    manage_io.create_directory(frames_dir, verbose=False)
    clear_stale_frames(frames_dir=frames_dir)
    path_x_values, path_y_values = generate_orbit_path()
    frame_angles = compute_frame_angles()
    for frame_index, angle in enumerate(frame_angles):
        draw_frame(
            frames_dir=frames_dir,
            frame_index=frame_index,
            angle=float(angle),
            path_x_values=path_x_values,
            path_y_values=path_y_values,
        )
    manage_log.log_action(
        title="Draw frames",
        outcome=manage_log.ActionOutcome.SUCCESS,
        message="Drew one period of the orbit.",
        notes={
            "frames": f"{NUM_FRAMES}",
            "period": f"{NUM_FRAMES / FRAMES_PER_SECOND:.2f} s at {FRAMES_PER_SECOND} fps",
        },
    )
    manage_figure.animate_frames_to_video(
        frames_dir=frames_dir,
        video_path=demo_dir / "orbit.mp4",
        frames_per_second=FRAMES_PER_SECOND,
    )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
