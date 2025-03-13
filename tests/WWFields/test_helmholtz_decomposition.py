## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from Loki.WWPlots import PlotUtils
from Loki.WWFields import FieldOperators, DeriveQuantities


## ###############################################################
## EXAMPLE VECTOR FIELDS
## ###############################################################
def genDivergenceVField(domain_bounds, num_cells):
  """Generate a divergence (curl-free) vector field."""
  domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
  grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
  sfield_qx = 2 * grid_x
  sfield_qy = 2 * grid_y
  sfield_qz = 2 * grid_z
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])

def genSolenoidalVField(domain_bounds, num_cells):
  """Generate a solenoidal (divergence-free) vector field."""
  domain_length = domain_bounds[1] - domain_bounds[0]
  domain = numpy.linspace(domain_bounds[0], domain_bounds[1], int(num_cells))
  constant = 2 * numpy.pi / domain_length
  grid_x, grid_y, grid_z = numpy.meshgrid(domain, domain, domain, indexing="ij")
  sfield_qx = -constant * grid_x * numpy.sin(constant * grid_x * grid_y)
  sfield_qy =  constant * grid_y * numpy.sin(constant * grid_x * grid_y)
  sfield_qz = numpy.zeros_like(grid_z)
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])

def genMixedVField(domain_bounds, num_cells):
  sfield_div = genDivergenceVField(domain_bounds, num_cells)
  sfield_sol = genSolenoidalVField(domain_bounds, num_cells)
  return sfield_div + sfield_sol


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def computeFieldFraction(bedges, pdf):
  nonzero_indices = numpy.where(pdf > 0)[0]
  if len(nonzero_indices) > 0:
    first_percent = bedges[nonzero_indices[0]]
    last_percent = bedges[nonzero_indices[-1]]
    return first_percent if first_percent == last_percent else (last_percent - first_percent)
  return 0.0

def plotVectorFieldSlice(ax, vfield_q, domain_bounds):
  _, num_cells_x, num_cells_y, _ = vfield_q.shape
  index_z = num_cells_x // 2 # middle slice in the z-direction
  grid_x, grid_y = numpy.meshgrid(
    numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_x),
    numpy.linspace(domain_bounds[0], domain_bounds[1], num_cells_y),
    indexing="xy"
  )
  sfield_q_magn_slice = FieldOperators.vfieldMagnitude(vfield_q[:,:,:,index_z])
  sfield_q_magn_min = numpy.min(sfield_q_magn_slice)
  sfield_q_magn_max = numpy.max(sfield_q_magn_slice)
  ax.imshow(
      sfield_q_magn_slice.T,
      origin = "lower",
      extent = [ domain_bounds[0], domain_bounds[1], domain_bounds[0], domain_bounds[1] ],
      cmap   = "viridis",
      alpha  = 0.7
  )
  ax.streamplot(
    grid_x,
    grid_y,
    vfield_q[0,:,:,index_z],
    vfield_q[1,:,:,index_z],
    color      = "black",
    arrowstyle = "->",
    linewidth  = 2.0,
    density    = 1.0,
    arrowsize  = 1.0,
    broken_streamlines = False,
  )
  ax.text(
    0.95, 0.95,
    f"magnitude: [{sfield_q_magn_min:.3f}, {sfield_q_magn_max:.3f}]",
    va="top", ha="right", transform=ax.transAxes,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
  )
  ax.set_xticks([])
  ax.set_yticks([])


## ###############################################################
## TESTING HELMHOLTZ DECOMPOSITION
## ###############################################################
def main():
  num_cells = 50
  domain_bounds = [ -1, 1 ]
  domain_length = domain_bounds[1] - domain_bounds[0]
  domain_size = (domain_length, domain_length, domain_length)
  list_vfields = [
    {"label": "divergence", "vfield": genDivergenceVField(domain_bounds, num_cells)},
    {"label": "solenoidal", "vfield": genSolenoidalVField(domain_bounds, num_cells)},
    {"label": "mixed",      "vfield": genMixedVField(domain_bounds, num_cells)},
  ]
  fig, axs = PlotUtils.initFigure(num_rows=3, num_cols=3, fig_aspect_ratio=(5,5))
  list_failed_vfields = []
  for vfield_index, vfield_entry in enumerate(list_vfields):
    vfield_name = vfield_entry["label"]
    vfield_q    = vfield_entry["vfield"]
    print(f"input: {vfield_name} field")
    vfield_q_div, vfield_q_sol   = DeriveQuantities.computeHelmholtzDecomposition(vfield_q, domain_size)
    sfield_check_q_diff          = FieldOperators.vfieldMagnitude((vfield_q - (vfield_q_div + vfield_q_sol)))
    sfield_check_div_is_sol_free = FieldOperators.vfieldMagnitude(FieldOperators.vfieldCurl(vfield_q_div))
    sfield_check_sol_is_div_free = FieldOperators.vfieldDivergence(vfield_q_sol)
    ave_q_diff     = numpy.median(numpy.abs(sfield_check_q_diff))
    ave_sol_in_div = numpy.median(numpy.abs(sfield_check_div_is_sol_free))
    ave_div_in_sol = numpy.median(numpy.abs(sfield_check_sol_is_div_free))
    std_q_diff     = numpy.std(numpy.abs(sfield_check_q_diff))
    std_sol_in_div = numpy.std(numpy.abs(sfield_check_div_is_sol_free))
    std_div_in_sol = numpy.std(numpy.abs(sfield_check_sol_is_div_free))
    bool_q_returned      = ave_q_diff < 0.5
    bool_div_is_sol_free = ave_sol_in_div < 0.5
    bool_sol_is_div_free = ave_div_in_sol < 0.5
    print(f"|q - (q_div + q_div)| / |q| = {ave_q_diff:.2f} +/- {std_q_diff:.2f}")
    print(f"curl(q_div) = {ave_sol_in_div:.2f} +/- {std_sol_in_div:.2f}")
    print(f"div(q_div) = {ave_div_in_sol:.2f} +/- {std_div_in_sol:.2f}")
    plotVectorFieldSlice(axs[vfield_index,0], vfield_q, domain_bounds)
    plotVectorFieldSlice(axs[vfield_index,1], vfield_q_div, domain_bounds)
    plotVectorFieldSlice(axs[vfield_index,2], vfield_q_sol, domain_bounds)
    axs[vfield_index,0].text(
      0.05, 0.05,
      f"input: {vfield_name} field",
      va="bottom", ha="left", transform=axs[vfield_index,0].transAxes,
      bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )
    axs[vfield_index,1].text(
      0.05, 0.05,
      "measured: divergence component",
      va="bottom", ha="left", transform=axs[vfield_index,1].transAxes,
      bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )
    axs[vfield_index,2].text(
      0.05, 0.05,
      "measured: solenoidal component",
      va="bottom", ha="left", transform=axs[vfield_index,2].transAxes,
      bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
    )
    if not(bool_q_returned and bool_div_is_sol_free and bool_sol_is_div_free):
      if not(bool_q_returned): print("Failed test. q_div + q_div =/= q")
      if not(bool_div_is_sol_free): print("Failed test. |curl(q_div)}| > 0")
      if not(bool_sol_is_div_free): print("Failed test. |div{q_curl)}| > 0")
      list_failed_vfields.append(vfield_name)
    else: print("Test passed successfully!")
    print(" ")
  PlotUtils.saveFigure(fig, "helmholtz_decomposition.png")
  assert len(list_failed_vfields) == 0, f"Test failed for the following vector field(s): {list_failed_vfields}"
  print("All tests passed successfully!")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST