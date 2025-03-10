## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy
from Loki.WWPlots import PlotUtils


## ###############################################################
## FUNCTION: HELMHOLTZ DECOMPOSITION
## ###############################################################
def computeHelmholtzDecomposition(vfield_q: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
  """
  Decomposes a 3D vector field into its solenoidal (divergence-free) and 
  compressive (curl-free) components using Helmholtz decomposition.

  Args:
    vfield_q (numpy.ndarray): Input velocity field of shape (3, num_cells_x, num_cells_y, num_cells_z),
              where 3 corresponds to (sfield_qx, sfield_qy, sfield_qz).

  Returns:
    tuple[numpy.ndarray, numpy.ndarray]: 
    - solenoidal component (3, num_cells_x, num_cells_y, num_cells_z)
    - compressive component (3, num_cells_x, num_cells_y, num_cells_z)
  """
  assert vfield_q.shape[0] == 3, "Input vector field must have shape (3, num_cells_x, num_cells_y, num_cells_z)"
  sfield_qx, sfield_qy, sfield_qz = vfield_q
  num_cells_x, num_cells_y, num_cells_z = sfield_qx.shape
  array_kx = numpy.fft.fftfreq(num_cells_x) * num_cells_x
  array_ky = numpy.fft.fftfreq(num_cells_y) * num_cells_y
  array_kz = numpy.fft.fftfreq(num_cells_z) * num_cells_z
  grid_kx, grid_ky, grid_kz = numpy.meshgrid(array_kx, array_ky, array_kz, indexing="ij")
  grid_k_magn = grid_kx**2 + grid_ky**2 + grid_kz**2
  grid_k_magn[0, 0, 0] = 1 # avoid division by zero
  sfield_hat_qx = numpy.fft.fftn(sfield_qx)
  sfield_hat_qy = numpy.fft.fftn(sfield_qy)
  sfield_hat_qz = numpy.fft.fftn(sfield_qz)
  # dot product: for compressive component
  sfield_q_dot_k = grid_kx * sfield_hat_qx + grid_ky * sfield_hat_qy + grid_kz * sfield_hat_qz
  ## compressive (curl-free) component
  sfield_hat_qx_comp = (sfield_q_dot_k / grid_k_magn) * grid_kx
  sfield_hat_qy_comp = (sfield_q_dot_k / grid_k_magn) * grid_ky
  sfield_hat_qz_comp = (sfield_q_dot_k / grid_k_magn) * grid_kz
  ## solenoidal (divergence-free) component
  sfield_hat_qx_sol = sfield_hat_qx - sfield_hat_qx_comp
  sfield_hat_qy_sol = sfield_hat_qy - sfield_hat_qy_comp
  sfield_hat_qz_sol = sfield_hat_qz - sfield_hat_qz_comp
  ## convert back to real space
  v_sol = numpy.stack([
    numpy.fft.ifftn(sfield_hat_qx_sol).real,
    numpy.fft.ifftn(sfield_hat_qy_sol).real,
    numpy.fft.ifftn(sfield_hat_qz_sol).real
  ])
  v_comp = numpy.stack([
    numpy.fft.ifftn(sfield_hat_qx_comp).real,
    numpy.fft.ifftn(sfield_hat_qy_comp).real,
    numpy.fft.ifftn(sfield_hat_qz_comp).real
  ])
  return v_sol, v_comp


## ###############################################################
## ANALYTIC TEST FIELDS
## ###############################################################
def genSolenoidalvfield_q(num_cells_x, num_cells_y, num_cells_z):
  """Generate a divergence-free (solenoidal) vector field: curl of a potential."""
  x, y, z = numpy.meshgrid(numpy.arange(num_cells_x), numpy.arange(num_cells_y), numpy.arange(num_cells_z), indexing="ij")
  sfield_qx =  numpy.sin(y) * numpy.cos(z)
  sfield_qy = -numpy.cos(x) * numpy.sin(z)
  sfield_qz =  numpy.sin(x) * numpy.cos(y)
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])
  
def genCompressivevfield_q(num_cells_x, num_cells_y, num_cells_z):
  """Generate a curl-free (compressive) vector field: gradient of a potential."""
  x, y, z = numpy.meshgrid(numpy.arange(num_cells_x), numpy.arange(num_cells_y), numpy.arange(num_cells_z), indexing="ij")
  sfield_qx = x
  sfield_qy = y
  sfield_qz = z
  return numpy.stack([ sfield_qx, sfield_qy, sfield_qz ])

def genMixedvfield_q(num_cells_x, num_cells_y, num_cells_z):
  sfield_sol  = genSolenoidalvfield_q(num_cells_x, num_cells_y, num_cells_z)
  sfield_comp = genCompressivevfield_q(num_cells_x, num_cells_y, num_cells_z)
  return sfield_sol + sfield_comp


## ###############################################################
## TESTING HELMHOLTZ DECOMPOSITION WITH ANALYTIC FIELDS
## ###############################################################
def main():
  num_cells_x, num_cells_y, num_cells_z = 64, 64, 64
  print(f"Testing Helmholtz decomposition on grid: {num_cells_x}x{num_cells_y}x{num_cells_z}")
  analytic_fields = {
    "Solenoidal Field"  : genSolenoidalvfield_q(num_cells_x, num_cells_y, num_cells_z),
    "Compressive Field" : genCompressivevfield_q(num_cells_x, num_cells_y, num_cells_z),
    "Mixed Field"       : genMixedvfield_q(num_cells_x, num_cells_y, num_cells_z)
  }
  fig, axs = PlotUtils.initFigure(num_rows=3, num_cols=3)
  for field_idx, (field_name, vfield_q) in enumerate(analytic_fields.items()):
    print(f"Testing {field_name}...")
    start_time = time.perf_counter()
    vfield_q_sol, vfield_q_comp = computeHelmholtzDecomposition(vfield_q)
    end_time = time.perf_counter()
    print(f"Decomposition completed in {end_time - start_time:.3f} seconds.")
    slice_idx = num_cells_z // 2
    for i, (label, data) in enumerate(zip(
        ["Original", "Solenoidal", "Compressive"],
        [vfield_q, vfield_q_sol, vfield_q_comp])
      ):
      ax = axs[i, field_idx]
      im = ax.imshow(data[0,:,slice_idx,:].T, origin="lower", cmap="coolwarm")
      ax.set_title(f"{label} {field_name}: $v_x$")
      fig.colorbar(im, ax=ax)
  PlotUtils.saveFigure(fig, "helmholtz_decomposition.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST