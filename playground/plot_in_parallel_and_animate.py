import numpy
from pathlib import Path
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager
from jormi.parallelism import independent_tasks

def render_sine_frame(
    output_dir  : str | Path,
    frame_index : int,
    num_frames  : int,
  ) -> str:
  x = numpy.linspace(0, 1, 1000)
  phase = frame_index / max(1, num_frames - 1)
  y = numpy.sin(2 * numpy.pi * (x - phase))
  fig, ax = plot_manager.create_figure()
  ax = plot_manager.cast_to_axis(ax)
  ax.plot(x, y, linewidth=2.0)
  ax.set_xlim(0, 1)
  ax.set_ylim(-1.1, 1.1)
  ax.set_xlabel(r"$x$")
  ax.set_ylabel(r"$\sin(2\pi(x - \phi))$")
  ax.set_title(rf"frame {frame_index+1}/{num_frames}: $\phi={phase:.3f}$")
  png_path = Path(output_dir) / f"frame_{frame_index:05d}.png"
  plot_manager.save_figure(fig, png_path, verbose=False)
  return str(png_path)

def main():
  num_frames = 300
  output_dir = Path(__file__).parent / "frames"
  io_manager.init_directory(output_dir)
  grouped_args = [
    (output_dir, frame_index, num_frames)
    for frame_index in range(num_frames)
  ]
  independent_tasks.run_in_parallel(
    func             = render_sine_frame,
    grouped_args     = grouped_args,
    timeout_seconds  = 60.0,
    show_progress    = True,
    enable_plotting  = True,
  )
  plot_manager.animate_png_to_mp4(
    frames_dir = output_dir,
    mp4_path   = output_dir / "animation_v3.mp4",
    fps        = 120,
  )

if __name__ == "__main__":
  main()

## .