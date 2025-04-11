## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import h5py
import numpy
from jormungandr.ww_io import file_manager


## ###############################################################
## FUNCTIONS
## ###############################################################
def read_grid_properties(file_path):
  file_manager.does_file_exist(file_path=file_path, raise_error=True)
  def _extract_properties(_h5file, dataset_name):
    return {
      str(key).split("'")[1].strip(): value
      for key, value in _h5file[dataset_name]
    }
  ## check that the file is the right type and has the right structure before proceeding
  properties = {}
  try:
    ## read datasets
    with h5py.File(file_path, "r") as h5file:
      properties["plasma_datasets"] = [
        dataset_name
        for dataset_name in h5file.keys()
        if any(
          dataset_name.startswith(prefix)
          for prefix in ("mag", "vel", "dens", "cur")
        )
      ]
      properties["int_scalars"]    = _extract_properties(h5file, "integer scalars")
      properties["int_properties"] = _extract_properties(h5file, "integer runtime parameters")
  except KeyError as exception:
    print(f"Error: The group {exception} was not found in: {file_path}.")
    return {}
  except Exception as exception:
    print(f"An unexpected error occurred: {exception}")
    return {}
  if len(properties["plasma_datasets"]) == 0: print(f"Warning: No plasma datasets found in: {file_path}")
  try:
    output_num    = properties["int_scalars"]["plotfilenumber"]
    dataset_names = properties["plasma_datasets"]
    num_blocks    = numpy.int32(properties["int_scalars"]["globalnumblocks"])
    num_blocks_x  = numpy.int32(properties["int_properties"]["iprocs"])
    num_blocks_y  = numpy.int32(properties["int_properties"]["jprocs"])
    num_blocks_z  = numpy.int32(properties["int_properties"]["kprocs"])
    num_cells_per_block_x = numpy.int32(properties["int_scalars"]["nxb"])
    num_cells_per_block_y = numpy.int32(properties["int_scalars"]["nyb"])
    num_cells_per_block_z = numpy.int32(properties["int_scalars"]["nzb"])
    num_cells_per_block   = num_cells_per_block_x * num_cells_per_block_y * num_cells_per_block_z
    num_cells             = num_blocks * num_cells_per_block
    return {
      "output_num"            : output_num,
      "dataset_names"         : dataset_names,
      "num_blocks"            : num_blocks,
      "num_blocks_x"          : num_blocks_x,
      "num_blocks_y"          : num_blocks_y,
      "num_blocks_z"          : num_blocks_z,
      "num_cells_per_block_x" : num_cells_per_block_x,
      "num_cells_per_block_y" : num_cells_per_block_y,
      "num_cells_per_block_z" : num_cells_per_block_z,
      "num_cells"             : num_cells,
    }
  except KeyError as missing_key:
    print(f"Error: Missing key `{missing_key}` in the extracted properties from: {file_path}")
    return {}

def reformat_flash_sfield(
    sfield              : numpy.ndarray,
    num_blocks          : tuple[int, int, int],
    num_cells_per_block : tuple[int, int, int],
  ):
  ## input Fortran-style field (column-major: z, y, x) with shape:
  ## [total_number_of_blocks, num_cells_per_block_z, num_cells_per_block_y, num_cells_per_block_x]
  ## where total_number_of_blocks = num_blocks_x * num_blocks_y * num_blocks_z
  ## reshape this field to separate the unified block-structure into its individual block components:
  ## total_number_of_blocks -> [num_blocks_z, num_blocks_y, num_blocks_x]
  sfield = sfield.reshape(
    num_blocks[2], num_blocks[1], num_blocks[0],
    num_cells_per_block[2], num_cells_per_block[1], num_cells_per_block[0]
  )
  ## interleave blocks with their cells
  sfield = numpy.transpose(sfield, (0, 3, 1, 4, 2, 5))
  ## merge block and cell dimensions
  sfield_sorted = sfield.reshape(
    num_blocks[2] * num_cells_per_block[2],
    num_blocks[1] * num_cells_per_block[1],
    num_blocks[0] * num_cells_per_block[0],
  )
  ## convert from Fortran-style [z, y, x] to C-style [x, y, z] cell ordering
  return sfield_sorted.transpose((2, 1, 0))


## END OF MODULE