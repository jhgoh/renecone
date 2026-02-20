#!/usr/bin/env python
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
from numba import njit
from typing import Dict, List, Optional, Sequence, Tuple

import sys
sys.path.append('python')
from ConeProfile import *

tol: float = 1e-7  # Numerical tolerance
_EXIT = 0
_BOUNCED_BACK = 1
_ON_SENSOR = 2
_BOUNCE_LIMIT = 3
_MIRROR = 10


@njit(cache=True)
def _propagate_cone_exit_code(
    x0: float,
    y0: float,
    z0: float,
    angle: float,
    mirror_ws: np.ndarray,
    mirror_hs: np.ndarray,
    mirror_lens: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    rmax: float,
    ymax: float,
    n_bounces: int = 20,
) -> int:
  rad = np.deg2rad(angle - 90)
  vx, vy, vz = np.cos(rad), np.sin(rad), 0.0

  for _ in range(n_bounces):
    bestR = np.inf
    bestX, bestY, bestZ = x0, y0, z0
    bestType = -1
    bestNx, bestNy, bestNz = 0.0, 0.0, 0.0

    for i in range(mirror_ws.shape[0]):
      n = mirror_lens[i]
      mw = mirror_ws[i, :n]
      mh = mirror_hs[i, :n]
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, mw, mh)
      if x.shape[0] > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = np.argmin(r)
        if r[irmin] < bestR:
          bestR = r[irmin]
          bestX, bestY, bestZ = x[irmin], y[irmin], z[irmin]
          bestType = _MIRROR
          bestNx, bestNy, bestNz = nx[irmin], ny[irmin], nz[irmin]

    x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, sx, sy)
    if x.shape[0] > 0:
      r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
      irmin = np.argmin(r)
      if r[irmin] < bestR:
        bestR = r[irmin]
        bestX, bestY, bestZ = x[irmin], y[irmin], z[irmin]
        bestType = _ON_SENSOR

    if bestType == -1 and abs(vy) > tol:
      t_back = (1.1 * ymax - y0) / vy
      if t_back > tol:
        x_back = x0 + t_back * vx
        z_back = z0 + t_back * vz
        w_back = np.hypot(x_back, z_back)
        if w_back <= 2.0 * rmax + tol:
          bestX, bestY, bestZ = x_back, 1.1 * ymax, z_back
          bestType = _BOUNCED_BACK

    if bestType == -1 and abs(vy) > tol:
      t_exit = -y0 / vy
      if t_exit > tol:
        x_exit = x0 + t_exit * vx
        z_exit = z0 + t_exit * vz
        w_exit = np.hypot(x_exit, z_exit)
        if w_exit <= 2.0 * rmax + tol:
          bestX, bestY, bestZ = x_exit, 0.0, z_exit
          bestType = _EXIT

    if bestType == -1:
      return _EXIT

    x0, y0, z0 = bestX, bestY, bestZ

    if bestType != _MIRROR:
      return bestType

    vx, vy, vz = reflect(vx, vy, vz, bestNx, bestNy, bestNz)

  return _BOUNCE_LIMIT


def _prepare_numba_geometry(
    mirrors: List[Dict[str, Sequence[float]]],
    sensor: Dict[str, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
  mws: List[np.ndarray] = []
  mhs: List[np.ndarray] = []
  lens: List[int] = []
  rmax = 0.0
  ymax = 0.0

  for mirror in mirrors:
    mw = np.abs(np.array(mirror['x'], dtype=np.float64))
    mh = np.array(mirror['y'], dtype=np.float64)
    mws.append(mw)
    mhs.append(mh)
    lens.append(len(mw))
    rmax = max(rmax, float(np.max(mw)))
    ymax = max(ymax, float(np.max(mh)))

  max_len = max(lens)
  mirror_ws = np.zeros((len(mws), max_len), dtype=np.float64)
  mirror_hs = np.zeros((len(mhs), max_len), dtype=np.float64)
  mirror_lens = np.array(lens, dtype=np.int64)

  for i, (mw, mh) in enumerate(zip(mws, mhs)):
    mirror_ws[i, :len(mw)] = mw
    mirror_hs[i, :len(mh)] = mh

  sx = np.abs(np.array(sensor['x'], dtype=np.float64))
  sy = np.array(sensor['y'], dtype=np.float64)
  rmax = max(rmax, float(np.max(sx)))
  ymax = max(ymax, float(np.max(sy)))

  return mirror_ws, mirror_hs, mirror_lens, sx, sy, rmax, ymax


def _scan_one_angle_cone(
    inc_angle: float,
    mirror_ws: np.ndarray,
    mirror_hs: np.ndarray,
    mirror_lens: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    height: float,
    radius: float,
    rmax: float,
    ymax: float,
    n_rays: int,
    seed: int,
) -> Tuple[int, int, int, int]:
  rng = np.random.default_rng(seed)
  n_pass, n_entr, n_esc, n_blim = 0, 0, 0, 0

  theta0s = rng.uniform(0, 2 * np.pi, n_rays)
  r0s = radius * np.sqrt(rng.uniform(0, 1, n_rays))
  x0s, z0s = r0s * np.cos(theta0s), r0s * np.sin(theta0s)

  for x0, z0 in zip(x0s, z0s):
    exit_code = _propagate_cone_exit_code(
        z0,
        height - 1,
        x0,
        inc_angle,
        mirror_ws,
        mirror_hs,
        mirror_lens,
        sx,
        sy,
        rmax,
        ymax,
    )
    if exit_code == _ON_SENSOR:
      n_pass += 1
    elif exit_code == _BOUNCED_BACK:
      n_entr += 1
    elif exit_code == _BOUNCE_LIMIT:
      n_blim += 1
    else:
      n_esc += 1

  return n_pass, n_entr, n_esc, n_blim


def _scan_one_angle_cone_from_args(
    args: Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, int, int]
) -> Tuple[int, int, int, int]:
  return _scan_one_angle_cone(*args)

@njit(cache=True)
def findSegments(
    x0: float, y0: float, z0: float,
    vx: float, vy: float, vz: float,
    mw: np.ndarray, mh: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """3D ray/surface intersections."""
  vw = np.hypot(vx, vz)

  dw = mw[1:] - mw[:-1]
  dh = mh[1:] - mh[:-1]
  nw, ny = -dh, dw

  dh2 = dh**2
  dw2 = dw**2

  hy0 = mh - y0
  k = mw[:-1] * hy0[1:] - mw[1:] * hy0[:-1]

  a = dh2 * (vw**2) - dw2 * (vy**2)
  b = dh2 * (vx * x0 + vz * z0) - k * vy * dw
  c = dh2 * (x0 * x0 + z0 * z0) - k**2

  t1 = np.full(dw.shape, np.inf)
  t2 = np.full(dw.shape, np.inf)
  
  is2pol = np.abs(a) > tol
  det = b**2 - a * c

  # a=0, b!=0 case: 1 polynomial
  mask = (~is2pol) & (np.abs(b) > tol)
  t1[mask] = t2[mask] = -c[mask] / b[mask] / 2

  # a!=0, det=0 case: 2nd order polynomial but 1 solution
  mask = is2pol & (np.abs(det) <= tol)
  t1[mask] = t2[mask] = -b[mask] / a[mask]

  # det > 0 case: two solutions. We take smaller one with positive solution
  mask = is2pol & (det > tol)
  rt = np.sqrt(det[mask])
  t1[mask] = (-b[mask] + rt) / a[mask]
  t2[mask] = (-b[mask] - rt) / a[mask]
  t1[t1 <= tol] = np.inf
  t2[t2 <= tol] = np.inf
  t1, t2 = np.minimum(t1, t2), np.maximum(t1, t2)

  x = x0 + t1 * vx
  y = y0 + t1 * vy
  z = z0 + t1 * vz
  w = np.hypot(x, z)

  s = ((w - mw[:-1]) * dw + (y - mh[:-1]) * dh) / (dw2 + dh2)
  branch = (dh * w - dw * y) - (dh * mw[:-1] - dw * mh[:-1])
  tol_branch = tol * (np.abs(dh) * w + np.abs(dw) * np.abs(y) + np.abs(dh * mw[:-1] - dw * mh[:-1]) + 1.0)
  hit1 = (t1 < np.inf) & (s >= -tol) & (s <= 1 + tol) & (np.abs(branch) <= tol_branch) & (w > tol)

  toSwap = (~hit1) & (t2 < np.inf)
  x[toSwap] = x0 + t2[toSwap] * vx
  y[toSwap] = y0 + t2[toSwap] * vy
  z[toSwap] = z0 + t2[toSwap] * vz
  w[toSwap] = np.hypot(x[toSwap], z[toSwap])

  s = ((w - mw[:-1]) * dw + (y - mh[:-1]) * dh) / (dw2 + dh2)
  branch = (dh * w - dw * y) - (dh * mw[:-1] - dw * mh[:-1])
  tol_branch = tol * (np.abs(dh) * w + np.abs(dw) * np.abs(y) + np.abs(dh * mw[:-1] - dw * mh[:-1]) + 1.0)
  hit = (s >= -tol) & (s <= 1 + tol) & (np.abs(branch) <= tol_branch) & (w > tol)

  nx = nw * x / w  # = nw * cos(phi)
  nz = nw * z / w  # = nw * sin(phi)

  return x[hit], y[hit], z[hit], nx[hit], ny[hit], nz[hit]

@njit(cache=True)
def getDist(
    x0: float, y0: float, z0: float,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    vx: float, vy: float, vz: float,
) -> float:
  """Distances from the ray origin to hits."""
  dx, dy, dz = x - x0, y - y0, z - z0
  vr = (vx**2 + vy**2 + vz**2) ** 0.5
  r = (vx * dx + vy * dy + vz * dz) / vr

  return r

@njit(cache=True)
def reflect(
    vx: float, vy: float, vz: float,
    nx: float, ny: float, nz: float,
) -> Tuple[float, float, float]:
  """Reflect a 3D ray about a normal."""
  nr = (nx**2 + ny**2 + nz**2) ** 0.5
  nx, ny, nz = nx / nr, ny / nr, nz / nr

  norm = vx * nx + vy * ny + vz * nz
  if norm > 0:
    nx, ny, nz = -nx, -ny, -nz
    norm = -norm

  ux = vx - 2 * norm * nx
  uy = vy - 2 * norm * ny
  uz = vz - 2 * norm * nz

  return ux, uy, uz  

def propagate(
    x0: float,
    y0: float,
    z0: float,
    angle: float,
    mirrors: List[Dict[str, Sequence[float]]],
    sensor: Optional[Dict[str, Sequence[float]]] = None,
    n_bounces: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
  """Trace one ray in the rotational geometry."""
  rad = np.deg2rad(angle - 90)  # initial light ray of theta=0 points downwards
  vx, vy, vz = np.cos(rad), np.sin(rad), 0  # initial direction; z-component is ignored through the screen
  xs, ys, zs = [x0], [y0], [z0]
  exit_type = 'bounce limit'

  rmin, rmax, ymax = None, None, None

  mws, mhs = [], []
  for mirror in mirrors:
    mw = np.abs(np.array(mirror['x'], dtype=np.float64))
    mh = np.array(mirror['y'], dtype=np.float64)
    mws.append(mw)
    mhs.append(mh)

    rmax = mw.max() if rmax is None else max(rmax, np.abs(mw.min()), mw.max())
    ymax = mh.max() if ymax is None else max(ymax, mh.max())

  if sensor:
    sx = np.abs(np.array(sensor['x'], dtype=np.float64))
    sy = np.array(sensor['y'], dtype=np.float64)

    rmax = sx.max() if rmax is None else max(rmax, np.abs(sx.min()), sx.max())
    ymax = sy.max() if ymax is None else max(ymax, sy.max())

  for _ in range(n_bounces):
    bestR, bestX, bestY, bestZ = None, None, None, None
    bestType = None
    bestTangent = None
    for mw, mh in zip(mws, mhs):
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, mw, mh)
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'mirror'
          bestTangent = [nx[irmin], ny[irmin], nz[irmin]]
    if sensor:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, sx, sy)
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'on sensor'
    # Add a virtual layer to pick up rays escaping backwards
    if bestType is not None:
      x0, y0, z0 = bestX, bestY, bestZ
    else:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([0, 2 * rmax]), np.array([1.1 * ymax, 1.1 * ymax]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, x0, y0, z0 = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'bounced back'
    if bestType is None:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([0, 2 * rmax]), np.array([0, 0]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, x0, y0, z0 = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'exit'

    xs.append(x0)
    ys.append(y0)
    zs.append(z0)

    if bestType != 'mirror':
      exit_type = bestType if bestType else 'bounced back'
      break

    vx, vy, vz = reflect(vx, vy, vz, *bestTangent)

  return np.array(xs), np.array(ys), np.array(zs), exit_type

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='3D Winston cone ray tracing scan')
  parser.add_argument('-o', '--output', required=True,
                      help='Path to the CSV file where scan results are stored')
  parser.add_argument('-q', '--quiet', action='store_true',
                      help='Silent mode: skip drawing figures')
  parser.add_argument('--height', type=float, default=1200,
                      help='Height of the cone (mm)')
  parser.add_argument('--width', type=float, default=1200,
                      help='Width of the cone (mm)')
  parser.add_argument('--din', type=float, default=1200,
                      help='Entrance diameter (mm)')
  parser.add_argument('--dout', type=float, default=460,
                      help='Exit diameter (mm)')
  parser.add_argument('--angle', type=float, default=20,
                      help='Cone opening angle (deg)')
  parser.add_argument('--sensor-curv', type=float, default=325,
                      help='Sensor curvature radius (mm)')
  parser.add_argument('--mirror-type', choices=['winston', 'planar'], default='winston',
                      help='Type of mirror profile used to build the concentrator')
  parser.add_argument('--n-rays-vis', type=int, default=51,
                      help='Number of rays to draw in the visualisation')
  parser.add_argument('--n-rays', type=int, default=10000,
                      help='Number of rays to trace per scan point')
  parser.add_argument('--inc-angle', type=float, default=20,
                      help='Incident angle for the visualised rays (deg)')
  parser.add_argument('--scan-min', type=float, default=0,
                      help='Minimum incident angle for the scan (deg)')
  parser.add_argument('--scan-max', type=float, default=90,
                      help='Maximum incident angle for the scan (deg)')
  parser.add_argument('--scan-steps', type=int, default=200,
                      help='Number of scan points between min and max angles')
  parser.add_argument('--jobs', type=int, default=1,
                      help='Number of processes for scan over incident angles (default: 1)')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  hep.style.use(hep.style.CMS)

  par_height = args.height
  par_width = args.width
  par_din = args.din
  par_dout = args.dout
  par_angle = args.angle
  par_sensor_curv = args.sensor_curv
  par_n_rays_vis = args.n_rays_vis
  par_n_rays = args.n_rays
  vis_inc_angle = args.inc_angle

  if args.mirror_type == 'winston':
    config = make_winston(par_din, par_dout, par_angle, par_width, par_height, sides=['right'])
  elif args.mirror_type == 'planar':
    config = make_planar(par_din, par_dout, par_angle, par_width, par_height, sides=['right'])
  else:
    raise ValueError(f"Unsupported mirror type: {args.mirror_type}")
  config['sensor'] = make_sensor(par_dout, sensor_curv=par_sensor_curv, n_points=32, sides=['right'])

  radius = config['din'] / 2 * 0.9
  mirror_ws, mirror_hs, mirror_lens, sx, sy, rmax_scan, ymax_scan = _prepare_numba_geometry(
      config['mirrors'], config['sensor']
  )

  # Warm-up JIT once to pay compile cost upfront.
  _ = _propagate_cone_exit_code(
      0.0,
      par_height - 1,
      0.0,
      vis_inc_angle,
      mirror_ws,
      mirror_hs,
      mirror_lens,
      sx,
      sy,
      rmax_scan,
      ymax_scan,
  )

  if not args.quiet:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plt.tight_layout(pad=2.0)
    thetas = np.linspace(0, 2 * np.pi, 100)

    rmax = 0
    for shape in config['mirrors']:
      x, y = shape['x'], shape['y']
      x, y = np.array(x), np.array(y)
      axes[0].plot(x, y, 'k')
      axes[0].plot(-x, y, 'k')

      xmin, xmax = x.min(), x.max()
      #axes[1].plot(xmin * np.cos(thetas), xmin * np.sin(thetas), 'k')
      axes[1].plot(xmax * np.cos(thetas), xmax * np.sin(thetas), 'k')

      rmax = max(np.hypot(x, y).max(), rmax)

    x, y = config['sensor']['x'], config['sensor']['y']
    x, y = np.array(x), np.array(y)
    axes[0].plot(x, y, 'g', linewidth=2)
    axes[0].plot(-x, y, 'g', linewidth=2)

    xmin, xmax = x.min(), x.max()
    axes[1].plot(xmin * np.cos(thetas), xmin * np.sin(thetas), 'g')
    axes[1].plot(xmax * np.cos(thetas), xmax * np.sin(thetas), 'g')

    axes[0].plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
    axes[0].plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

    span = config['din'] / 2 * 0.9
    for x0 in np.linspace(-span, span, par_n_rays_vis):
      z0 = 0
      xs, ys, zs, exit_type = propagate(z0, par_height - 1, x0, vis_inc_angle,
                                         config['mirrors'], sensor=config['sensor'])
      color = {'exit': 'b', 'bounced back': 'r', 'bounce limit': 'r', 'on sensor': 'g'}
      axes[0].set_xlabel('x (mm)')
      axes[0].set_ylabel('y (mm)')
      axes[0].plot(xs, ys, color[exit_type] + '-', linewidth=0.5)

      axes[1].set_xlabel('x (mm)')
      axes[1].set_ylabel('z (mm)')
      axes[1].plot(xs, zs, color[exit_type] + '-', linewidth=0.5)

    axes[0].set_xlim(-rmax, rmax)
    axes[0].set_ylim(-0.2 * rmax, 1.2 * rmax)
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].set_xlim(-rmax, rmax)
    axes[1].set_ylim(-rmax, rmax)
    axes[1].set_aspect('equal', adjustable='box')
    plt.show()

  inc_angles = np.linspace(args.scan_min, args.scan_max, args.scan_steps)
  frac_pass = np.zeros(len(inc_angles))
  frac_entr = np.zeros(len(inc_angles))
  frac_esc = np.zeros(len(inc_angles))
  frac_blim = np.zeros(len(inc_angles))

  n_jobs = max(1, args.jobs)
  seeds = np.random.SeedSequence(12345).generate_state(len(inc_angles), dtype=np.uint64)

  if n_jobs == 1:
    iterator = zip(inc_angles, seeds)
    for i, (inc_angle, seed) in enumerate(tqdm(iterator, total=len(inc_angles))):
      n_pass, n_entr, n_esc, n_blim = _scan_one_angle_cone(
          inc_angle,
          mirror_ws,
          mirror_hs,
          mirror_lens,
          sx,
          sy,
          par_height,
          radius,
          rmax_scan,
          ymax_scan,
          par_n_rays,
          int(seed),
      )
      frac_pass[i] = n_pass / par_n_rays
      frac_entr[i] = n_entr / par_n_rays
      frac_esc[i] = n_esc / par_n_rays
      frac_blim[i] = n_blim / par_n_rays
  else:
    max_workers = min(n_jobs, os.cpu_count() or 1)
    tasks = [
        (float(inc_angle), mirror_ws, mirror_hs, mirror_lens, sx, sy, par_height, radius, rmax_scan, ymax_scan, par_n_rays, int(seed))
        for inc_angle, seed in zip(inc_angles, seeds)
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      results = list(tqdm(executor.map(_scan_one_angle_cone_from_args, tasks), total=len(tasks)))

    for i, (n_pass, n_entr, n_esc, n_blim) in enumerate(results):
      frac_pass[i] = n_pass / par_n_rays
      frac_entr[i] = n_entr / par_n_rays
      frac_esc[i] = n_esc / par_n_rays
      frac_blim[i] = n_blim / par_n_rays

  if not args.quiet:
    plt.plot(inc_angles, frac_pass, 'b.-', label='on sensor')
    plt.plot(inc_angles, frac_entr, 'r.-', label='bounced back')
    plt.plot(inc_angles, frac_esc, 'k.-', label='escaped (unclassified)')
    plt.plot(inc_angles, frac_blim, 'm.-', label='bounce limit')
    plt.xlabel('Indicent angle (deg)')
    plt.ylabel('Fraction')
    plt.legend()
    plt.show()

  with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['inc_angle_deg', 'fraction_on_sensor', 'fraction_bounced_back', 'fraction_escaped_unclassified', 'fraction_bounce_limit'])
    for angle, frac_on_sensor, frac_bounced, frac_escaped, frac_bounce_limit in zip(inc_angles, frac_pass, frac_entr, frac_esc, frac_blim):
      writer.writerow([angle, frac_on_sensor, frac_bounced, frac_escaped, frac_bounce_limit])
