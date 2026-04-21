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
│   └── jormi/           # package root (ww_ = "working with")
│       ├── ww_fields/   # compute spectra, decompositions, etc of 2D/3D scalar+vector fields
│       ├── ww_arrays/   # array operations (norms, masking, probability density functions)
│       ├── ww_types/    # type hints and runtime type checking
│       ├── ww_data/     # fitting and interpolating data series
│       ├── ww_fns/      # function decorators and parallel dispatch
│       ├── ww_io/       # file I/O (CSV, JSON), logging, shell commands
│       ├── ww_jobs/     # HPC (PBS and SLURM) job submission
│       ├── ww_plots/    # mpl figures, styling, colormaps, annotations
│       ├── ww_dicts.py  # dict helpers
│       ├── ww_lists.py  # list helpers
│       └── ww_stats.py  # statistics
├── utests/              # unit tests
├── vtests/              # validation tests
├── demos/               # example scripts
├── pyproject.toml       # project metadata and dependencies
├── uv.lock              # pinned dependency versions
└── README.md            # this file
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
