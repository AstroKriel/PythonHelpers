## { MODULE

##
## === DEPENDENCIES
##

import numpy
import cmasher
from pathlib import Path
from jormi.ww_data import fit_data
from jormi.ww_plots import plot_manager, annotate_axis
from jormi.ww_fields import generate_fields, compute_spectra

##
## === MAIN PROGRAM
##


def main():
    num_cells = 100
    slope = -2
    k_bin_centers = None
    power_spectra = []
    sfield = None
    for _ in range(5):
        sfield = generate_fields.generate_powerlaw_sfield(
            num_cells=num_cells,
            alpha_perp=-slope,
        )
        k_bin_centers, spectrum_1d = compute_spectra.compute_1d_power_spectrum(field=sfield)
        power_spectra.append(spectrum_1d)
    fig, axs_grid = plot_manager.create_figure(num_cols=2)
    axs_list = list(numpy.squeeze(axs_grid))
    axs_list[0].fill_between(
        k_bin_centers,
        numpy.percentile(power_spectra, 16, axis=0),
        numpy.percentile(power_spectra, 84, axis=0),
        color="blue",
        alpha=0.3,
    )
    axs_list[0].plot(
        k_bin_centers,
        numpy.percentile(power_spectra, 50, axis=0),
        color="blue",
        marker="o",
        ms=5,
        linestyle="-",
        linewidth=1,
    )
    x_values = numpy.logspace(-1, 3, 10)
    line_intercept = 10**fit_data.get_linear_intercept(
        slope=slope,
        x_ref=1,
        y_ref=-12,
    )
    rotate_deg = fit_data.get_line_angle(
        slope=slope,
        domain_bounds=(0, 2, -14, -11),
        domain_aspect_ratio=6 / 4,
    )
    annotate_axis.overlay_curve(
        ax=axs_list[0],
        x_values=x_values,
        y_values=line_intercept * numpy.power(x_values, slope),
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    annotate_axis.add_text(
        ax=axs_list[0],
        x_pos=0.5,
        y_pos=0.75,
        label=rf"$\mathcal{{P}}(k) \propto k^{{{slope:.2f}}}$",
        x_alignment="center",
        y_alignment="center",
        rotate_deg=rotate_deg,
    )
    axs_list[0].set_xscale("log")
    axs_list[0].set_yscale("log")
    axs_list[0].set_xlabel("wavenumber")
    axs_list[0].set_ylabel("power spectrum")
    axs_list[0].set_xlim([1, num_cells])
    axs_list[0].set_ylim([1e-14, 1e-11])
    if sfield is not None:
        axs_list[1].imshow(sfield[:, :, num_cells // 2], cmap="cmr.rainforest")
    axs_list[1].set_xticks([])
    axs_list[1].set_yticks([])
    script_path = Path(__file__).parent
    fig_path = script_path / "generated_powerlaw_sfield.png"
    plot_manager.save_figure(fig, fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
