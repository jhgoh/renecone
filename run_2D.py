#!/usr/bin/env python
import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
from numba import njit
from typing import Dict, List, Optional, Sequence, Tuple

import sys
sys.path.append('python')
from ConeProfile import *
from trace_outcomes import EXIT_COLOR

tol: float = 1e-7

@njit(cache=True)
def findSegments(
    x0: float, y0: float,
    vx: float, vy: float,
    mx: np.ndarray, my: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Ray/polyline intersections."""
  dmx = mx[1:] - mx[:-1]
  dmy = my[1:] - my[:-1]
  nx, ny = -dmy, dmx

  ss = vx * nx + vy * ny
  ok = np.abs(ss) > tol

  qx = mx[:-1] - x0
  qy = my[:-1] - y0

  t = np.full(ss.shape, np.inf)
  u = np.full(ss.shape, np.inf)

  t[ok] = (qx[ok] * nx[ok] + qy[ok] * ny[ok]) / ss[ok]
  u[ok] = (-qx[ok] * vy + qy[ok] * vx) / ss[ok]

  hit = ok & (t >= tol) & (u >= -tol) & (u <= 1 + tol)
  if not np.any(hit):
    return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

  x = x0 + t * vx
  y = y0 + t * vy

  return x[hit], y[hit], nx[hit], ny[hit]

@njit(cache=True)
def getDist(
    x0: float, y0: float,
    x: np.ndarray, y: np.ndarray,
    vx: float, vy: float,
) -> np.ndarray:
  """Project hit offsets onto the ray direction."""
  dx, dy = x - x0, y - y0
  r = (vx * dx + vy * dy) / np.hypot(vx, vy)

  return r

@njit(cache=True)
def reflect(vx: float, vy: float, nx: float, ny: float) -> Tuple[float, float]:
  """Reflect a 2D ray about a normal."""
  nr = np.hypot(nx, ny)
  nx, ny = nx / nr, ny / nr

  norm = vx * nx + vy * ny
  if norm > 0:
    nx, ny = -nx, -ny
    norm = -norm

  ux = vx - 2 * norm * nx
  uy = vy - 2 * norm * ny

  return ux, uy

def propagate(
    x0: float,
    y0: float,
    angle: float,
    mirrors: List[Dict[str, Sequence[float]]],
    sensor: Optional[Dict[str, Sequence[float]]] = None,
    n_bounces: int = 20,
) -> Tuple[np.ndarray, np.ndarray, str]:
  """Trace one ray through the optics."""
  rad = np.deg2rad(angle - 90)
  vx, vy = np.cos(rad), np.sin(rad)
  xs, ys = [x0], [y0]
  exit_type = 'bounce limit'

  xmin, xmax, ymax = None, None, None

  mxs, mys = [], []
  for mirror in mirrors:
    mx = np.array(mirror['x'], dtype=np.float64)
    my = np.array(mirror['y'], dtype=np.float64)
    mxs.append(mx)
    mys.append(my)

    xmin = mx.min() if xmin is None else min(xmin, mx.min())
    xmax = mx.max() if xmax is None else max(xmax, mx.max())
    ymax = my.max() if ymax is None else max(ymax, my.max())

  if sensor:
    sx = np.array(sensor['x'], dtype=np.float64)
    sy = np.array(sensor['y'], dtype=np.float64)

    xmin = sx.min() if xmin is None else min(xmin, sx.min())
    xmax = sx.max() if xmax is None else max(xmax, sx.max())
    ymax = sy.max() if ymax is None else max(ymax, sy.max())

  while n_bounces >= 0:
    bestR, bestX, bestY = None, None, None
    bestType = None
    bestTangent = None
    for mx, my in zip(mxs, mys):
      x, y, nx, ny = findSegments(x0, y0, vx, vy, mx, my)
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, bestX, bestY = r[irmin], x[irmin], y[irmin]
          bestType = 'mirror'
          bestTangent = [nx[irmin], ny[irmin]]
    if sensor:
      x, y, nx, ny = findSegments(x0, y0, vx, vy, sx, sy)
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, bestX, bestY = r[irmin], x[irmin], y[irmin]
          bestType = 'on sensor'
    # Add a virtual layer to pick up rays escaping backwards
    if bestType is None:
      x, y, nx, ny = findSegments(x0, y0, vx, vy,
                                  np.array([xmin, xmax], dtype=np.float64),
                                  np.array([ymax, ymax], dtype=np.float64))
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, x0, y0 = r[irmin], x[irmin], y[irmin]
          bestType = 'bounced back'
    else:
      x0, y0 = bestX, bestY
    if bestType is None:
      x, y, nx, ny = findSegments(x0, y0, vx, vy,
                                  np.array([xmin, xmax], dtype=np.float64),
                                  np.array([0, 0], dtype=np.float64))
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR is None or r[irmin] < bestR:
          bestR, x0, y0 = r[irmin], x[irmin], y[irmin]
          bestType = 'exit'

    xs.append(x0)
    ys.append(y0)

    if bestType == 'mirror':
      n_bounces -= 1
      vx, vy = reflect(vx, vy, *bestTangent)
    else:
      exit_type = bestType if bestType else 'bounced back'
      break

  return np.array(xs), np.array(ys), exit_type

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='2D Winston cone ray tracing scan')
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
  parser.add_argument('--n-rays-vis', type=int, default=101,
                      help='Number of rays to draw in the visualisation')
  parser.add_argument('--n-rays', type=int, default=10000,
                      help='Number of rays to trace per scan point')
  parser.add_argument('--inc-angle', type=float, default=-30,
                      help='Incident angle for the visualised rays (deg)')
  parser.add_argument('--scan-min', type=float, default=0,
                      help='Minimum incident angle for the scan (deg)')
  parser.add_argument('--scan-max', type=float, default=90,
                      help='Maximum incident angle for the scan (deg)')
  parser.add_argument('--scan-steps', type=int, default=200,
                      help='Number of scan points between min and max angles')

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
    config = make_winston(par_din, par_dout, par_angle, par_width, par_height)
  elif args.mirror_type == 'planar':
    config = make_planar(par_din, par_dout, par_angle, par_width, par_height)
  else:
    raise ValueError(f"Unsupported mirror type: {args.mirror_type}")
  config['sensor'] = make_sensor(par_dout, sensor_curv=par_sensor_curv)

  radius = config['din'] / 2 * 0.9

  if not args.quiet:
    rmax = 0
    for shape in config['mirrors']:
      x, y = shape['x'], shape['y']
      x, y = np.array(x), np.array(y)
      plt.plot(x, y, 'k')

      rmax = max(np.hypot(x, y).max(), rmax)

    x, y = config['sensor']['x'], config['sensor']['y']
    plt.plot(x, y, 'g', linewidth=2)

    plt.plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
    plt.plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

    for x0 in np.linspace(-radius, radius, par_n_rays_vis):
      xs, ys, exit_type = propagate(x0, par_height - 1, vis_inc_angle,
                                    config['mirrors'], sensor=config['sensor'])
      color = EXIT_COLOR
      plt.xlabel('x (mm)')
      plt.ylabel('y (mm)')
      plt.plot(xs, ys, color[exit_type] + '-', linewidth=0.5)

    plt.xlim(-rmax, rmax)
    plt.ylim(-0.2 * rmax, 1.2 * rmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

  inc_angles = np.linspace(args.scan_min, args.scan_max, args.scan_steps)
  frac_pass = np.zeros(len(inc_angles))
  frac_entr = np.zeros(len(inc_angles))
  frac_exit = np.zeros(len(inc_angles))
  frac_blim = np.zeros(len(inc_angles))
  for i, inc_angle in enumerate(tqdm(inc_angles)):
    n_pass, n_entr, n_exit, n_blim = 0, 0, 0, 0
    for x0 in np.random.uniform(-radius, radius, par_n_rays):
      _, _, exit_type = propagate(x0, par_height - 1, inc_angle,
                                  config['mirrors'], sensor=config['sensor'])
      if exit_type == 'on sensor':
        n_pass += 1
      elif exit_type == 'bounced back':
        n_entr += 1
      elif exit_type == 'exit':
        n_exit += 1
      elif exit_type == 'bounce limit':
        n_blim += 1
    frac_pass[i] = n_pass / par_n_rays
    frac_entr[i] = n_entr / par_n_rays
    frac_exit[i] = n_exit / par_n_rays
    frac_blim[i] = n_blim / par_n_rays

  if not args.quiet:
    plt.plot(inc_angles, frac_pass, 'b.-', label='on sensor')
    plt.plot(inc_angles, frac_entr, 'r.-', label='bounced back')
    plt.plot(inc_angles, frac_exit, 'k.-', label='exit')
    plt.plot(inc_angles, frac_blim, 'm.-', label='bounce limit')
    plt.xlabel('Indicent angle (deg)')
    plt.ylabel('Fraction')
    plt.legend()
    plt.show()

  with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['inc_angle_deg', 'fraction_on_sensor', 'fraction_bounced_back', 'fraction_exit', 'fraction_bounce_limit'])
    for angle, frac_on_sensor, frac_bounced, frac_exited, frac_bounce_limit in zip(inc_angles, frac_pass, frac_entr, frac_exit, frac_blim):
      writer.writerow([angle, frac_on_sensor, frac_bounced, frac_exited, frac_bounce_limit])

