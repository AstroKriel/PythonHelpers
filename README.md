# jormi (Jormungandr; the World Serpent)

jormi is a Python utility library for computing all things plasma physics, including:
- decomposition (e.g., Helmholtz and Frenet-Serret basis)
- power spectra
- probability density functions
- differential operators (e.g., curl, divergence, gradient)
- field slices and interpolation

as well as general utilities for managing figures, fitting data, type-safe I/O, and runtime type checking. It serves as a base layer for many of my simulation-specific libraries that add interfaces on top of this.

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

Run any script through the managed environment with:

```bash
uv run path/to/script.py
```

## File structure

```
jormi/
├── src/
│   └── jormi/                  # package root (ww_ = "working with")
│       ├── ww_fields/          # 2D/3D scalar+vector fields; differential operators, spectra, decomposition
│       ├── ww_arrays/          # array operations (norms, masking, probability density functions)
│       ├── ww_types/           # type hints and runtime type checking
│       ├── ww_data/            # fitting and interpolating data series
│       ├── ww_fns/             # function decorators and parallel dispatch
│       ├── ww_io/              # file I/O (CSV, JSON), logging, shell commands
│       ├── ww_jobs/            # HPC (PBS) job submission
│       ├── ww_plots/           # mpl figures, styling, colormaps, annotations
│       ├── ww_dicts.py         # dict helpers
│       ├── ww_lists.py         # list helpers
│       └── ww_stats.py         # statistics
├── utests/                     # unit tests
├── vtests/                     # validation tests
├── demos/                      # example scripts
├── pyproject.toml              # project metadata and dependencies
├── uv.lock                     # pinned dependency versions
└── README.md                   # this file
```

## Running tests

Run the suite of unit tests:

```bash
uv run pytest utests/
```

Run the suite of validation tests:

```bash
uv run vtests/run.py
```

## License

See [LICENSE.md](./LICENSE.md).
