# renecone

Utilities for generating concentrator shapes and running lightweight ray-tracing
studies using Python. The code was originally written as a compact sandbox for
testing Winston cones but the helper functions are flexible enough to extend to
other light-guiding geometries.

## Repository layout

The repository contains two executable examples backed by a small geometry
library:

- **`run_2D.py`** – builds planar concentrators or Winston cones in 2D and
  traces sample rays through the profile. Core geometric routines are
  accelerated with [Numba](https://numba.pydata.org/) so that scans across many
  incident angles complete quickly.
- **`run_3Dcone.py`** – demonstrates a simple 3D extension of the cone profile
  and performs similar ray‑tracing studies using matplotlib's 3D plotting
  utilities.
- **`python/ConeProfile.py`** – shared utilities that generate the mirrors and
  optional sensor surfaces used by both example scripts.

## Prerequisites

Create and activate an environment that provides the numeric and plotting
libraries used by the scripts. The commands below set up a new conda
environment, but any workflow that installs the dependencies is fine:

```bash
conda create -n renecone -c conda-forge numpy numba matplotlib mplhep tqdm
conda activate renecone
```

If you prefer `pip`, install the same packages into a virtual environment.

## Quick start

Run one of the scripts to render the shapes and trace a few sample rays:

```bash
python run_2D.py     # 2D profile
python run_3Dcone.py # 3D demonstration
```

On headless systems, or when running on a batch farm, you can use the
non-interactive matplotlib backend:

```bash
MPLBACKEND=Agg python run_2D.py
MPLBACKEND=Agg python run_3Dcone.py
```

The scripts will open a window (or generate an off‑screen figure) displaying the
constructed geometry and ray paths. The last part of each script performs a scan
over incidence angles and plots the transmission/entrance fractions, giving a
quick way to evaluate performance.

## Customisation tips

Key parameters such as input/output diameters, mirror angles and sensor
curvature are defined in the `__main__` sections of the example scripts. Update
them directly or expose them through your own argument parser to explore
different configurations. For more involved studies, you can:

- Swap `make_winston` for `make_planar` (or a custom helper) to test alternative
  shapes.
- Adjust `par_n_rays` and `par_n_rays_vis` to trade off runtime versus plotting
  density.
- Inspect the `propagate` function to see how ray histories are stored if you
  wish to collect additional diagnostics.

The helper functions in `ConeProfile.py` are pure Python/Numpy routines, so they
are easy to adapt when experimenting with new concentrator designs.
