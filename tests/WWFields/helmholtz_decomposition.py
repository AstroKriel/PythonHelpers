## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy
from Loki.WWPlots import PlotUtils


## ###############################################################
## FUNCTION: HELMHOLTZ DECOMPOSITION
## ###############################################################
def computeHelmholtzDecomposition(vfield: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
  """
  Decomposes a 3D vector field into its solenoidal (divergence-free) and 
  compressive (curl-free) components using Helmholtz decomposition.

  Args:
    vfield (numpy.ndarray): Input velocity field of shape (3, Nx, Ny, Nz),
              where 3 corresponds to (vx, vy, vz).

  Returns:
    tuple[numpy.ndarray, numpy.ndarray]: 
    - Solenoidal component (3, Nx, Ny, Nz)
    - Compressive component (3, Nx, Ny, Nz)
  """
  assert vfield.shape[0] == 3, "Input vector field must have shape (3, Nx, Ny, Nz)"
  vx, vy, vz = vfield
  Nx, Ny, Nz = vx.shape
  kx = numpy.fft.fftfreq(Nx) * Nx
  ky = numpy.fft.fftfreq(Ny) * Ny
  kz = numpy.fft.fftfreq(Nz) * Nz
  kx, ky, kz = numpy.meshgrid(kx, ky, kz, indexing="ij")
  k_squared = kx**2 + ky**2 + kz**2
  ## avoid division by zero
  k_squared[0, 0, 0] = 1
  v_hat_x = numpy.fft.fftn(vx)
  v_hat_y = numpy.fft.fftn(vy)
  v_hat_z = numpy.fft.fftn(vz)
  # dot product: for compressive component
  k_dot_v = kx * v_hat_x + ky * v_hat_y + kz * v_hat_z
  ## compressive (curl-free) component
  v_hat_comp_x = (k_dot_v / k_squared) * kx
  v_hat_comp_y = (k_dot_v / k_squared) * ky
  v_hat_comp_z = (k_dot_v / k_squared) * kz
  ## solenoidal (divergence-free) component
  v_hat_sol_x = v_hat_x - v_hat_comp_x
  v_hat_sol_y = v_hat_y - v_hat_comp_y
  v_hat_sol_z = v_hat_z - v_hat_comp_z
  v_sol = numpy.stack([
    numpy.fft.ifftn(v_hat_sol_x).real,
    numpy.fft.ifftn(v_hat_sol_y).real,
    numpy.fft.ifftn(v_hat_sol_z).real
  ])
  v_comp = numpy.stack([
    numpy.fft.ifftn(v_hat_comp_x).real,
    numpy.fft.ifftn(v_hat_comp_y).real,
    numpy.fft.ifftn(v_hat_comp_z).real
  ])
  return v_sol, v_comp


## ###############################################################
## ANALYTIC TEST FIELDS
## ###############################################################
def genSolenoidalVField(Nx, Ny, Nz):
  """Generate a divergence-free (solenoidal) vector field: curl of a potential."""
  x, y, z = numpy.meshgrid(numpy.arange(Nx), numpy.arange(Ny), numpy.arange(Nz), indexing="ij")
  vx =  numpy.sin(y) * numpy.cos(z)
  vy = -numpy.cos(x) * numpy.sin(z)
  vz =  numpy.sin(x) * numpy.cos(y)
  return numpy.stack([vx, vy, vz])
  
def genCompressiveVField(Nx, Ny, Nz):
  """Generate a curl-free (compressive) vector field: gradient of a potential."""
  x, y, z = numpy.meshgrid(numpy.arange(Nx), numpy.arange(Ny), numpy.arange(Nz), indexing="ij")
  vx = x
  vy = y
  vz = z
  return numpy.stack([vx, vy, vz])

def genMixedVField(Nx, Ny, Nz):
  sol_field = genSolenoidalVField(Nx, Ny, Nz)
  comp_field = genCompressiveVField(Nx, Ny, Nz)
  return sol_field + comp_field


## ###############################################################
## TESTING HELMHOLTZ DECOMPOSITION WITH ANALYTIC FIELDS
## ###############################################################
def main():
  Nx, Ny, Nz = 64, 64, 64
  print(f"Testing Helmholtz decomposition on grid: {Nx}x{Ny}x{Nz}")
  analytic_fields = {
    "Solenoidal Field"  : genSolenoidalVField(Nx, Ny, Nz),
    "Compressive Field" : genCompressiveVField(Nx, Ny, Nz),
    "Mixed Field"       : genMixedVField(Nx, Ny, Nz)
  }
  fig, axs = PlotUtils.initFigure(num_rows=3, num_cols=3)
  for field_idx, (field_name, vfield) in enumerate(analytic_fields.items()):
    print(f"Testing {field_name}...")
    start_time = time.perf_counter()
    vfield_sol, vfield_comp = computeHelmholtzDecomposition(vfield)
    end_time = time.perf_counter()
    print(f"Decomposition completed in {end_time - start_time:.3f} seconds.")
    slice_idx = Nz // 2
    for i, (label, data) in enumerate(zip(
        ["Original", "Solenoidal", "Compressive"],
        [vfield, vfield_sol, vfield_comp])
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