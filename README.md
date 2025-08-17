# renecone

Utilities for generating concentrator shapes using Python.

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

The script will open a window (or generate an off‑screen figure) displaying the constructed geometry and ray paths.
