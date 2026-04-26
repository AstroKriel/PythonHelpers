# jormi (Jormungandr; the World Serpent)

jormi is a utility library for computing all kinds of MHD turbulence related statistics, including:
- vector field decompositions (e.g., Helmholtz and Frenet-Serret basis)
- 1D and 3D power spectra
- 1D and joint probability density functions
- differential operators (e.g., curl, divergence)

as well as providing general utilities for fitting data series, managing figures, type-safe I/O, and runtime type checking. It serves as a base layer for many of my simulation-specific libraries that add interfaces on top.

## Getting setup

jormi is typically used as a submodule within the [Asgard](https://github.com/AstroKriel/Asgard) project. You can, however, clone this repository directly for standalone development:

```bash
git clone git@github.com:AstroKriel/PythonHelpers.git jormi  # clone and rename
cd jormi
uv sync
```

To make jormi importable from other projects in editable mode:

```bash
uv pip install -e .
```

## File structure

```
jormi/
├── src/
│   └── jormi/                  # package root (ww_ = "working with")
│       ├── ww_arrays/          # array operations (norms, masking, PDFs, spectra)
│       │   ├── farrays_3d/     # Fourier-array ops (spectra, decompositions, differential operators)
│       │   ├── compute_array_stats.py
│       │   ├── mask_2d_arrays.py
│       │   └── smooth_2d_arrays.py
│       ├── ww_data/            # fitting and interpolating data series
│       ├── ww_fields/          # scalar and vector field abstractions
│       │   ├── fields_2d/      # 2D field models and domain
│       │   └── fields_3d/      # 3D field models, domain, operators, spectra, decompositions
│       ├── ww_fns/             # function decorators and parallel dispatch
│       ├── ww_io/              # file I/O (CSV, JSON), logging, shell commands
│       ├── ww_jobs/            # HPC job submission
│       │   ├── pbs_manager/    # PBS job scripts and queue interface
│       │   └── slurm_manager/  # SLURM job scripts and queue interface
│       ├── ww_plots/           # matplotlib figures, styling, colormaps, annotations
│       │   └── color_palettes/ # discrete, sequential, and diverging palette builders
│       ├── ww_types/           # type hints, enums, and positional types
│       ├── ww_validation/      # runtime validation for arrays, types, enums, and box positions
│       ├── ww_dicts.py         # dict helpers
│       ├── ww_lists.py         # list helpers
│       └── ww_stats.py         # statistics helpers
├── utests/                     # unit tests (mirroring src/ layout)
│   ├── ww_arrays/
│   ├── ww_fields/
│   ├── ww_fns/
│   ├── ww_io/
│   ├── ww_jobs/
│   ├── ww_types/
│   ├── ww_validation/
│   ├── test_dicts.py
│   └── test_lists.py
├── vtests/                     # validation tests
│   ├── ww_arrays/
│   ├── ww_data/
│   ├── ww_fields/
│   └── run_all.py
├── pyproject.toml              # project metadata and dependencies
├── uv.lock                     # pinned dependency versions
└── README.md                   # this file
```

## Run test suites

Run the suite of unit tests:

```bash
uv run pytest
```

Run the suite of validation tests:

```bash
uv run vtests/run_all.py
```

## License

See [LICENSE.md](./LICENSE.md).
