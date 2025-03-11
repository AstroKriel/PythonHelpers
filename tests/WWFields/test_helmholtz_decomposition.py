## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWData import ComputePDFs
from Loki.WWPlots import PlotUtils
from Loki.WWFields import FieldOperators, DeriveQuantities


## ###############################################################
## EXAMPLE VECTOR FIELDS
## ###############################################################
def genSolenoidalVField(domain_bounds, num_cells):
  """Generate a solenoidal (divergence-free) vector field."""
  domain_component = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
  grid_x, grid_y, grid_z = numpy.meshgrid(domain_component, domain_component, domain_component, indexing="ij")
  sfield_qx = numpy.zeros_like(grid_x)
  sfield_qy = numpy.zeros_like(grid_y)
  sfield_qz = grid_y
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])

def genCompressiveVField(domain_bounds, num_cells):
  """Generate a compressive (curl-free) vector field."""
  domain_component = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
  grid_x, grid_y, grid_z = numpy.meshgrid(domain_component, domain_component, domain_component, indexing="ij")
  sfield_qx = 2 * grid_x
  sfield_qy = 2 * grid_y
  sfield_qz = 2 * grid_z
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])

def genMixedVField(domain_bounds, num_cells):
  sfield_sol  = genSolenoidalVField(domain_bounds, num_cells)
  sfield_comp = genCompressiveVField(domain_bounds, num_cells)
  return sfield_sol + sfield_comp


## ###############################################################
## TESTING HELMHOLTZ DECOMPOSITION
## ###############################################################
def main():
  num_cells = 1e2
  num_bins  = 50
  domain_bounds = [ -1, 1 ]
  domain_length = domain_bounds[1] - domain_bounds[0]
  domain_size = (domain_length, domain_length, domain_length)
  dict_vfields = [
    {"label": "solenoidal",  "vfield": genSolenoidalVField(domain_bounds, num_cells),  "color": "red"},
    {"label": "compressive", "vfield": genCompressiveVField(domain_bounds, num_cells), "color": "forestgreen"},
    {"label": "mixed",       "vfield": genMixedVField(domain_bounds, num_cells),       "color": "royalblue"},
  ]
  fig, axs = PlotUtils.initFigure(num_rows=2, num_cols=1)
  list_failed_vfields = []
  for vfield_entry in dict_vfields:
    vfield_name  = vfield_entry["label"]
    vfield_q     = vfield_entry["vfield"]
    vfield_color = vfield_entry["color"]
    vfield_q_comp, vfield_q_sol = DeriveQuantities.computeHelmholtzDecomposition(vfield_q, domain_size)
    sfield_magn_q         = FieldOperators.vfieldMagnitude(vfield_q)
    sfield_percent_comp   = FieldOperators.vfieldMagnitude(numpy.abs((vfield_q - vfield_q_comp) / sfield_magn_q))
    sfield_percent_sol    = FieldOperators.vfieldMagnitude(numpy.abs((vfield_q - vfield_q_sol) / sfield_magn_q))
    bedges_comp, pdf_comp = ComputePDFs.compute1DPDF(sfield_percent_comp.flatten(), num_bins=num_bins, bedge_extend_factor=0.5)
    bedges_sol, pdf_sol   = ComputePDFs.compute1DPDF(sfield_percent_sol.flatten(), num_bins=num_bins, bedge_extend_factor=0.5)
    axs[0].step(bedges_comp, pdf_comp, where="pre", lw=2, color=vfield_color)
    axs[1].step(bedges_sol, pdf_sol, where="pre", lw=2, color=vfield_color, label=vfield_name)
    nonzero_indices_comp = numpy.where(pdf_comp > 0)[0]
    if len(nonzero_indices_comp) > 0:
      first_nonzero = bedges_comp[nonzero_indices_comp[0]]
      last_nonzero = bedges_comp[nonzero_indices_comp[-1]]
      if first_nonzero == last_nonzero:
        fraction_comp = bedges_comp[0]
      else: fraction_comp = (last_nonzero - first_nonzero) / (bedges_comp[-1] - bedges_comp[0])
    else: fraction_comp = 0.0
    nonzero_indices_sol = numpy.where(pdf_sol > 0)[0]
    if len(nonzero_indices_sol) > 0:
      first_nonzero = bedges_sol[nonzero_indices_sol[0]]
      last_nonzero = bedges_sol[nonzero_indices_sol[-1]]
      if first_nonzero == last_nonzero:
        fraction_sol = bedges_sol[0]
      else: fraction_sol = (last_nonzero - first_nonzero) / (bedges_sol[-1] - bedges_sol[0])
    else: fraction_sol = 0.0
    print(f"{vfield_name} field")
    print(f"> compressive percentage: {100*fraction_comp:.2f}%")
    print(f"> solenoidal percentage: {100*fraction_sol:.2f}%")
    if 100 * (fraction_comp + fraction_sol) < 90: list_failed_vfields.append(vfield_name)
  axs[0].text(0.95, 0.95, "Compressive", ha="right", va="top", transform=axs[0].transAxes)
  axs[1].text(0.95, 0.95, "Solenoidal", ha="right", va="top", transform=axs[1].transAxes)
  axs[0].set_ylim([0, 1.1*numpy.max(pdf_comp)])
  axs[1].set_ylim([0, 1.1*numpy.max(pdf_sol)])
  axs[1].legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)
  PlotUtils.saveFigure(fig, "helmholtz_decomposition.png")
  assert len(list_failed_vfields) == 0, f"Test failed for the following vector field(s): {list_failed_vfields}"
  print("Test passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST