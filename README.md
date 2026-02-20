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
- **`run_3Dcone.py`** – rotationally symmetric 3D cone/Winston demonstration
  with ray-tracing scans.
- **`run_3Dhex.py`** – hexagonal entrance/exit concentrator using faceted side
  panels. Side shape is controlled by intermediate knot positions and scale
  factors so you can run panel-optimization studies.
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
python run_3Dcone.py -o results_3d.csv     # rotational 3D model
python run_3Dhex.py -o results_hex.csv      # hexagonal faceted 3D model
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
MPLBACKEND=Agg python run_3Dhex.py
```

The scripts will open a window (or generate an off‑screen figure) displaying the
constructed geometry and ray paths. The last part of each script performs a scan
over incidence angles and plots the transmission/entrance fractions, giving a
quick way to evaluate performance.

### Helpful command-line options

`run_2D.py` and `run_3Dcone.py` expose the following shared arguments:

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
- `--mirror-type` – choose between Winston (`winston`) and planar (`planar`)
  mirror profiles (default: Winston).
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


`run_3Dhex.py` uses the common scan controls (`-o`, `-q`, `--height`, `--din`, `--dout`, `--n-rays-vis`, `--n-rays`, `--inc-angle`, `--scan-min`, `--scan-max`, `--scan-steps`) and adds:

- `--jobs` – number of worker processes for the scan over incident angles. Use values greater than 1 to parallelise the hex scan across CPU cores (default: `1`).

- `--y-knots` – comma-separated y positions in normalized height `[0, 1]` where
  intermediate ring sections are created (default: `0.25,0.5,0.75`).
- `--scale-knots` – comma-separated radial multipliers at the same knot
  positions (default: `1.0,1.0,1.0`). Endpoints are fixed to 1.0 so entry/exit
  apertures remain at `din`/`dout`.

## Customisation tips

Key parameters such as input/output diameters, mirror angles and sensor
curvature are defined in the `__main__` sections of the example scripts. Update
them directly or expose them through your own argument parser to explore
different configurations. For more involved studies, you can:

- Pass `--mirror-type planar` to preview a simple flat-mirror concentrator in
  place of the default Winston profile. For bespoke geometries, edit the script
  to call your own helper routine.
- For hexagonal entry/exit studies with multiple flat side panels, use
  `python run_3Dhex.py` and tune `--y-knots` / `--scale-knots` to adjust
  intermediate panel positions and widths.
- Adjust `par_n_rays` and `par_n_rays_vis` to trade off runtime versus plotting
  density.
- Inspect the `propagate` function to see how ray histories are stored if you
  wish to collect additional diagnostics.

The helper functions in `ConeProfile.py` are pure Python/Numpy routines, so they
are easy to adapt when experimenting with new concentrator designs.
