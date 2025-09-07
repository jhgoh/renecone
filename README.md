# renecone

Utilities for generating concentrator shapes using Python.

The repository contains a single script, `run_2D.py`, which can build planar
concentrators or Winston cones and trace sample rays through the geometry. The
ray tracer is intentionally lightweight, making it easy to experiment with
different parameters or prototype new designs. Core geometric routines are
accelerated with [Numba](https://numba.pydata.org/) for faster ray-tracing
scans.

## Setup

Create and activate an Anaconda environment:

```bash
conda create -n renecone -c conda-forge numpy numba matplotlib tqdm
conda activate renecone
```

## Usage

Run the script to render the shapes and trace a few sample rays:

```bash
python run_2D.py
```

On headless systems, you can use the non-interactive backend:

```bash
MPLBACKEND=Agg python run_2D.py
```

The script will open a window (or generate an off‑screen figure) displaying the
constructed geometry and ray paths. Parameters such as input/output diameters,
mirror angles and sensor curvature can be adjusted in the `__main__` section of
`run_2D.py` to explore different configurations.
