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

Run one of the scripts to render the shapes and trace a few sample rays.
Both entry points require an output path where the scan summary will be
written:

```bash
python run_2D.py -o results_2d.csv         # 2D profile
python run_3Dcone.py -o results_3d.csv     # 3D demonstration
```

The CSV files contain the incident angle (in degrees) together with the
fraction of rays that reach the sensor or are reflected back out of the
entrance aperture. Re-running the script overwrites the file, so provide a
unique name when exploring multiple configurations.

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

### Helpful command-line options

Both scripts expose the same set of arguments:

- `-o / --output` (required) – destination CSV file for the scan summary.
- `-q / --quiet` – skip the matplotlib visualisations and run only the scan
  (default: disabled).
- `--height` – height of the cone in millimetres (default: 1200 mm).
- `--width` – width of the cone in millimetres (default: 1200 mm).
- `--din` – entrance diameter in millimetres (default: 1200 mm).
- `--dout` – exit diameter in millimetres (default: 460 mm).
- `--angle` – cone opening angle in degrees (default: 20°).
- `--sensor-curv` – radius of curvature for the optional sensor surface
  (default: 325 mm).
- `--n-rays-vis` – number of rays to draw in the preview figure (defaults: 101
  for the 2D script, 51 for the 3D script).
- `--n-rays` – number of rays traced per scan point, higher values reduce
  statistical noise (default: 10,000 rays).
- `--inc-angle` – incident angle used in the preview plot (defaults: -30° for
  2D, 20° for 3D).
- `--scan-min` – minimum incident angle in degrees for the scan (default: 0°).
- `--scan-max` – maximum incident angle in degrees for the scan (default: 90°).
- `--scan-steps` – number of scan points between the minimum and maximum angles
  (default: 200 steps).

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
