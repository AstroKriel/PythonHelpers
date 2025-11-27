## { MODULE

##
## === DEPENDENCIES
##

import numpy

##
## === FUNCTIONS
##


def _apply_2d_convolution(
    data: numpy.ndarray,
    smoothing_kernel: numpy.ndarray,
) -> numpy.ndarray:
    num_kernel_rows, num_kernel_cols = smoothing_kernel.shape
    num_pad_rows = num_kernel_rows // 2
    num_pad_cols = num_kernel_cols // 2
    padded_data = numpy.pad(
        data,
        ((num_pad_rows, num_pad_rows), (num_pad_cols, num_pad_cols)),
        mode="wrap",
    )
    num_data_rows, num_data_cols = data.shape
    output = numpy.zeros((num_data_rows, num_data_cols), dtype=numpy.float64)
    for index_row in range(num_data_rows):
        for index_col in range(num_data_cols):
            data_subset = padded_data[
                index_row:index_row + num_kernel_rows,
                index_col:index_col + num_kernel_cols,
            ]
            output[index_row, index_col] = numpy.sum(data_subset * smoothing_kernel)
    return output


def _define_2d_gaussian_kernel(
    size: int,
    sigma: float,
) -> numpy.ndarray:
    x_values = numpy.linspace(-(size // 2), size // 2, size)
    y_values = numpy.linspace(-(size // 2), size // 2, size)
    grid_x, grid_y = numpy.meshgrid(x_values, y_values)
    smoothing_kernel = numpy.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    smoothing_kernel /= numpy.sum(smoothing_kernel)
    return smoothing_kernel


def smooth_2d_array(
    data: numpy.ndarray,
    sigma: float,
) -> numpy.ndarray:
    kernel_size = int(6 * sigma) + 1
    smoothing_kernel = _define_2d_gaussian_kernel(kernel_size, sigma)
    smoothed_data = _apply_2d_convolution(data, smoothing_kernel)
    return smoothed_data


## } MODULE
