## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy

try:
  from mpi4py import MPI
  IS_MPI_AVAILABLE = True
  MPI_WORLD     = MPI.COMM_WORLD
  MPI_RANK      = MPI_WORLD.Get_rank()
  MPI_NUM_PROCS = MPI_WORLD.Get_size()
except ImportError:
  IS_MPI_AVAILABLE = False
  MPI_WORLD     = None
  MPI_RANK      = 0
  MPI_NUM_PROCS = 1


## ###############################################################
## FUNCTIONS
## ###############################################################
def compute_3d_power_spectrum(
    field   : numpy.ndarray,
    use_mpi : bool = False
  ) -> numpy.ndarray:
  """Computes the power spectrum of an arbitrary-dimensional field."""
  assert len(field.shape) >= 3, "Field should have at least 3 spatial dimensions."
  fft_field = numpy.fft.fftshift(numpy.fft.fftn(field, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))
  spectrum_local = numpy.sum(
    numpy.power(numpy.abs(fft_field), 2),
    axis = tuple(range(len(field.shape) - 3))
  )
  if use_mpi and IS_MPI_AVAILABLE:
    spectrum_global = numpy.empty_like(spectrum_local)
    MPI_WORLD.Allreduce(spectrum_local, spectrum_global, op=MPI.SUM)
    ## avoid redundant operations by only having the root process return the final global result
    return spectrum_global if MPI_WORLD.Get_rank() == 0 else None
  else: return spectrum_local

def perform_spherical_integration(
    spectrum_3d : numpy.ndarray,
    use_mpi     : bool = False
  ) -> tuple[numpy.ndarray, numpy.ndarray]:
  """Integrates a 3D power spectrum over spherical shells of constant k."""
  num_k_modes = spectrum_3d.shape[0] // 2
  k_bin_edges    = numpy.linspace(1, num_k_modes, num_k_modes+1)
  k_modes     = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
  k_center    = numpy.array([(_ki_size-1)/2.0 for _ki_size in spectrum_3d.shape], dtype=float)
  grid_kz, grid_ky, grid_kx = numpy.indices(spectrum_3d.shape)
  grid_k_magn    = numpy.sqrt((grid_kx - k_center[0])**2 + (grid_ky - k_center[1])**2 + (grid_kz - k_center[2])**2)
  bin_indices    = numpy.digitize(grid_k_magn, k_bin_edges)
  spectrum_local = numpy.bincount(bin_indices.ravel(), weights=spectrum_3d.ravel(), minlength=num_k_modes+1)[1:]
  if use_mpi and IS_MPI_AVAILABLE:
    spectrum_global = numpy.empty_like(spectrum_local)
    MPI_WORLD.Allreduce(spectrum_local, spectrum_global, op=MPI.SUM)
    ## avoid redundant operations by only having the root process return the final global result
    return k_modes, spectrum_global if MPI_WORLD.Get_rank() == 0 else None, None
  return k_modes, spectrum_local

def compute_1d_power_spectrum(
    field   : numpy.ndarray,
    use_mpi : bool = False
  ) -> tuple[numpy.ndarray, numpy.ndarray]:
  """Computes the full power spectrum including radial integration."""
  spectrum_3d = compute_3d_power_spectrum(field, use_mpi)
  if spectrum_3d is None: return None, None
  k_modes, spectrum_1d = perform_spherical_integration(spectrum_3d, use_mpi)
  if (k_modes is None) or (spectrum_1d is None): return None, None
  return k_modes, spectrum_1d


## END OF MODULE