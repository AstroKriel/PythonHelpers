## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy as np
import time
from Loki.WWPlots import PlotUtils

## ###############################################################
## FUNCTION: HELMHOLTZ DECOMPOSITION
## ###############################################################
def computeHelmholtzDecomposition(v_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """
  Decomposes a 3D vector field into its solenoidal (divergence-free) and 
  compressive (curl-free) components using Helmholtz decomposition.

  Args:
    v_field (np.ndarray): Input velocity field of shape (3, Nx, Ny, Nz),
              where 3 corresponds to (vx, vy, vz).

  Returns:
    tuple[np.ndarray, np.ndarray]: 
    - Solenoidal component (3, Nx, Ny, Nz)
    - Compressive component (3, Nx, Ny, Nz)
  """
  assert v_field.shape[0] == 3, "Input vector field must have shape (3, Nx, Ny, Nz)"
  vx, vy, vz = v_field
  Nx, Ny, Nz = vx.shape
  kx = np.fft.fftfreq(Nx) * Nx
  ky = np.fft.fftfreq(Ny) * Ny
  kz = np.fft.fftfreq(Nz) * Nz
  kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
  k_squared = kx**2 + ky**2 + kz**2
  ## avoid division by zero
  k_squared[0, 0, 0] = 1
  v_hat_x = np.fft.fftn(vx)
  v_hat_y = np.fft.fftn(vy)
  v_hat_z = np.fft.fftn(vz)
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
  v_sol = np.stack([
    np.fft.ifftn(v_hat_sol_x).real,
    np.fft.ifftn(v_hat_sol_y).real,
    np.fft.ifftn(v_hat_sol_z).real
  ])
  v_comp = np.stack([
    np.fft.ifftn(v_hat_comp_x).real,
    np.fft.ifftn(v_hat_comp_y).real,
    np.fft.ifftn(v_hat_comp_z).real
  ])
  return v_sol, v_comp


## ###############################################################
## ANALYTIC TEST FIELDS
## ###############################################################
def genSolenoidalVField(Nx, Ny, Nz):
  """Generate a divergence-free (solenoidal) vector field: curl of a potential."""
  x, y, z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
  vx =  np.sin(y) * np.cos(z)
  vy = -np.cos(x) * np.sin(z)
  vz =  np.sin(x) * np.cos(y)
  return np.stack([vx, vy, vz])
  
def genCompressiveVField(Nx, Ny, Nz):
  """Generate a curl-free (compressive) vector field: gradient of a potential."""
  x, y, z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
  vx = x
  vy = y
  vz = z
  return np.stack([vx, vy, vz])

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
  for field_idx, (field_name, v_field) in enumerate(analytic_fields.items()):
    print(f"Testing {field_name}...")
    start_time = time.perf_counter()
    v_sol, v_comp = computeHelmholtzDecomposition(v_field)
    end_time = time.perf_counter()
    print(f"Decomposition completed in {end_time - start_time:.3f} seconds.")
    slice_idx = Nz // 2
    for i, (label, data) in enumerate(zip(
        ["Original", "Solenoidal", "Compressive"],
        [v_field, v_sol, v_comp])
      ):
      ax = axs[i, field_idx]
      im = ax.imshow(data[0, :, slice_idx, :].T, origin="lower", cmap="coolwarm")
      ax.set_title(f"{label} {field_name}: $v_x$")
      fig.colorbar(im, ax=ax)
  PlotUtils.saveFigure(fig, "helmholtz_decomposition.png")


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF TEST