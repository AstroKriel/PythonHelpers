## { V-TEST

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
from jormi.ww_arrays import _filter_spectra
from jormi.ww_arrays.farrays_3d import filter_farrays
from jormi.ww_io import manage_log
from jormi.ww_plots import manage_figure, style_figure

##
## === BAND-PASS WINDOW COMPARISON
##

## (window, plot color); tukey needs a taper_width, passed separately
_WINDOWS_AND_COLORS: tuple[tuple[str, str], ...] = (
    ("tophat", "black"),
    ("gaussian", "firebrick"),
    ("tukey", "steelblue"),
)


class TestBandpassFilterWindows:
    """
    A hard top-hat k-space mask is an exact shell decomposition: it sums back to the
    original field exactly, but its real-space kernel is a long-range sinc. Band-limiting
    a compact, non-negative feature with it produces ringing that persists far from the
    feature, where the true field is exactly zero. A gaussian window has no hard edge, so
    it rings far less, but it also has no hard stop: it always leaks weight from outside
    [k_min, k_max]. A tukey window gets both: exactly zero outside the band (like the
    tophat), with a smooth, continuous-slope taper at the edges instead of a hard step
    (so it rings about as little as the gaussian). This is the concrete difference
    between "decomposition" and "filter" hiding behind the same k_min/k_max interface.
    """

    def __init__(
        self,
    ):
        self.resolution: tuple[int, int, int] = (128, 128, 128)
        self.k_min: float = 4.0
        self.k_max: float = 16.0
        self.taper_width: float = 3.0
        ## cells this far from the spike, along the sampled line, are scored as "far field":
        ## where the true field is exactly zero, so any residual amplitude is pure ringing
        self.far_field_min_distance: int = 15
        ## the tophat's far-field ringing must be clearly worse than every smooth window's,
        ## but "clearly worse" is not the same threshold for both: gaussian is C-infinity
        ## smooth (all derivatives continuous), so its real-space kernel decays fast and it
        ## suppresses ringing by orders of magnitude; tukey is only C1 (value and slope
        ## continuous, not curvature), so it suppresses ringing by degree, not by orders of
        ## magnitude. Holding tukey to the gaussian's bar would just mean picking a wider
        ## taper_width until it passes, at the cost of eroding the flat top that is the
        ## whole reason to reach for tukey over gaussian in the first place.
        self.min_ringing_ratio_by_window: dict[str, float] = {
            "gaussian": 3.0,
            "tukey": 1.5,
        }

    def run(
        self,
    ) -> None:
        spike_sarray = self._make_spike()
        filtered_by_window = {
            window: filter_farrays.compute_bandpass_filtered_sarray(
                sarray_3d=spike_sarray,
                resolution_3d=self.resolution,
                k_min=self.k_min,
                k_max=self.k_max,
                window=window,
                taper_width=(self.taper_width if window == "tukey" else None),
            )
            for window, _ in _WINDOWS_AND_COLORS
        }
        figure = self._plot_comparison(filtered_by_window)
        file_path = Path(__file__).parent / "bandpass_filter_windows.png"
        manage_figure.save_figure(
            figure=figure,
            figure_path=file_path,
        )
        ## the near-peak dip is comparable across windows; what actually differs is
        ## whether the ringing persists far from the spike, where the true field is
        ## exactly zero everywhere. RMS amplitude out there isolates that difference.
        far_field_rms_by_window = {
            window: self._far_field_rms(filtered) for window, filtered in filtered_by_window.items()
        }
        tophat_rms = far_field_rms_by_window["tophat"]
        ringing_ratios = {
            window: tophat_rms / max(rms, 1e-12)
            for window, rms in far_field_rms_by_window.items()
            if window != "tophat"
        }
        failures = {
            window: ratio
            for window, ratio in ringing_ratios.items()
            if not (ratio > self.min_ringing_ratio_by_window[window])
        }
        assert not failures, (
            f"expected the tophat window's far-field ringing to clear each smooth window's"
            f" own threshold ({self.min_ringing_ratio_by_window}) at the same nominal band;"
            f" got far-field RMS={far_field_rms_by_window}, ratios below threshold={failures}."
        )
        manage_log.log_action(
            title="Band-pass filter window comparison",
            outcome=manage_log.ActionOutcome.SUCCESS,
            message="The tophat window rings measurably more than the smooth windows, far from the spike.",
            notes={
                "far-field RMS": {window: f"{rms:.3e}" for window, rms in far_field_rms_by_window.items()},
                "ratio vs tophat": {window: f"{ratio:.2f}" for window, ratio in ringing_ratios.items()},
            },
        )

    def _make_spike(
        self,
    ) -> NDArray[Any]:
        """A single-cell spike: the sharpest possible non-negative feature."""
        sarray = numpy.zeros(self.resolution)
        center = tuple(num_cells // 2 for num_cells in self.resolution)
        sarray[center] = 1.0
        return sarray

    def _far_field_rms(
        self,
        filtered_sarray: NDArray[Any],
    ) -> float:
        """RMS amplitude at cells far enough from the spike that the true field is exactly
        zero there; any amplitude that survives is ringing, not signal. Scored over the
        full 3D volume (not just one line), so this is not an artifact of the sampled cut."""
        center = numpy.array([num_cells // 2 for num_cells in self.resolution])
        grid_indices = numpy.indices(self.resolution)
        distance_from_spike = numpy.linalg.norm(
            grid_indices - center.reshape(3, 1, 1, 1),
            axis=0,
        )
        far_field_mask = distance_from_spike >= self.far_field_min_distance
        return float(numpy.sqrt(numpy.mean(numpy.square(filtered_sarray[far_field_mask]))))

    def _plot_comparison(
        self,
        filtered_by_window: dict[str, NDArray[Any]],
    ) -> Any:
        figure, panel_grid = manage_figure.create_figure(
            num_panel_rows=1,
            num_panel_cols=2,
            panel_aspect_ratio=1.0,
        )
        mask_panel, profile_panel = panel_grid[0]
        ## left: all three masks share [k_min, k_max], but only tukey has both a hard
        ## stop (like the tophat) and a smooth edge (like the gaussian)
        k_values = numpy.linspace(0.0, 24.0, 400)
        for window, color in _WINDOWS_AND_COLORS:
            mask_panel.plot(
                k_values,
                _filter_spectra._compute_k_mask(
                    k_magnitude=k_values,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    window=window,
                    taper_width=(self.taper_width if window == "tukey" else None),
                ),
                color=color,
                label=window,
            )
        mask_panel.set_xlabel(r"$|\vec{k}|$")
        mask_panel.set_ylabel("mask amplitude")
        mask_panel.legend()
        ## right: the real-space consequence of that same nominal band, along a line
        ## through the spike; the spike itself is 0 everywhere except the center cell
        center = tuple(num_cells // 2 for num_cells in self.resolution)
        line_slice = (slice(None), center[1], center[2])
        x_indices = numpy.arange(self.resolution[0]) - center[0]
        profile_panel.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        for window, color in _WINDOWS_AND_COLORS:
            profile_panel.plot(
                x_indices,
                filtered_by_window[window][line_slice],
                color=color,
                label=f"{window}-filtered",
            )
        profile_panel.set_xlim(-32, 32)
        profile_panel.set_xlabel("cell index from spike")
        profile_panel.set_ylabel("filtered amplitude")
        profile_panel.legend()
        return figure


##
## === ENTRY POINT
##

if __name__ == "__main__":
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    style_figure.set_figure_params()
    test = TestBandpassFilterWindows()
    test.run()

## } V-TEST
