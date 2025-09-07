# renecone

Utilities for generating concentrator shapes using Python.

The repository contains two example scripts:

- **`run_2D.py`** – builds planar concentrators or Winston cones in 2D and
  traces sample rays through the profile. The ray tracer is intentionally
  lightweight, making it easy to experiment with different parameters or
  prototype new designs. Core geometric routines are accelerated with
  [Numba](https://numba.pydata.org/) for faster ray-tracing scans.
- **`run_3Dcone.py`** – demonstrates a simple 3D extension of the cone profile
  and performs similar ray‑tracing studies using matplotlib's 3D plotting
  utilities.

Shared geometry helpers such as `make_planar`, `make_winston` and
`make_sensor` live in `python/ConeProfile.py`.

## Setup

Create and activate an Anaconda environment:

```bash
conda create -n renecone -c conda-forge numpy numba matplotlib mplhep tqdm
conda activate renecone
```

## Usage

Run one of the scripts to render the shapes and trace a few sample rays:

```bash
python run_2D.py     # 2D profile
python run_3Dcone.py # 3D demonstration
```

On headless systems, you can use the non-interactive backend:

```bash
MPLBACKEND=Agg python run_2D.py
MPLBACKEND=Agg python run_3Dcone.py
```

The scripts will open a window (or generate an off‑screen figure) displaying
the constructed geometry and ray paths. Parameters such as input/output
diameters, mirror angles and sensor curvature can be adjusted in the
`__main__` sections of `run_2D.py` or `run_3Dcone.py` to explore different
configurations.
