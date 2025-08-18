# renecone

Utilities for generating concentrator shapes using Python.

The repository contains a single script, `makeShapes.py`, which can build
planar concentrators or Winston cones and trace sample rays through the
geometry.  The ray tracer is intentionally lightweight, making it easy to
experiment with different parameters or to prototype new designs.

## Setup

Install the required packages:

```bash
pip install numpy matplotlib
```

## Usage

Run the script to render the shapes and trace a few sample rays:

```bash
python makeShapes.py
```

On headless systems, you can use the non-interactive backend:

```bash
MPLBACKEND=Agg python makeShapes.py
```

The script will open a window (or generate an off‑screen figure) displaying the
constructed geometry and ray paths.  Parameters such as input/output diameters,
mirror angles and PMT curvature can be adjusted in the `__main__` section of
`makeShapes.py` to explore different configurations.
