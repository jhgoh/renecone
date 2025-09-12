#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
from numba import njit
from typing import Dict, List, Optional, Sequence, Tuple

import sys
sys.path.append('python')
from ConeProfile import *

tol: float = 1e-7

@njit(cache=True)
def findSegments(x0: float, y0: float, vx: float, vy: float,
                 mx: np.ndarray, my: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  dmx = mx[1:] - mx[:-1]
  dmy = my[1:] - my[:-1]

  ss = vx * dmy - vy * dmx
  ok = np.abs(ss) > tol

  qx = mx[:-1] - x0
  qy = my[:-1] - y0

  t = np.full(ss.shape, np.inf)
  u = np.full(ss.shape, np.inf)

  t[ok] = (qx[ok] * dmy[ok] - qy[ok] * dmx[ok]) / ss[ok]
  u[ok] = (qx[ok] * vy - qy[ok] * vx) / ss[ok]

  hit = ok & (t >= tol) & (u >= -tol) & (u <= 1 + tol)
  if not np.any(hit):
    return (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))

  x = x0 + t * vx
  y = y0 + t * vy
  nx, ny = -dmy, dmx

  return x[hit], y[hit], nx[hit], ny[hit]

@njit(cache=True)
def getDist(x0: float, y0: float,
            x: np.ndarray, y: np.ndarray,
            vx: float, vy: float) -> np.ndarray:
  dx = x - x0
  dy = y - y0
  r = (vx * dx + vy * dy) / np.hypot(vx, vy)

  return r

@njit(cache=True)
def reflect(vx: float, vy: float, nx: float, ny: float) -> Tuple[float, float]:
  nr = np.hypot(nx, ny)
  nx, ny = nx/nr, ny/nr

  norm = vx * nx + vy * ny
  if norm > 0:
    nx, ny = -nx, -ny
    norm = -norm

  ux = vx - 2 * norm * nx
  uy = vy - 2 * norm * ny

  return ux, uy

def propagate(x0: float, y0: float, angle: float,
              mirrors: List[Dict[str, Sequence[float]]],
              sensor: Optional[Dict[str, Sequence[float]]] = None,
              n_bounces: int = 20) -> Tuple[np.ndarray, np.ndarray, str]:
  rad = np.deg2rad(angle-90)
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

    xmin = mx.min() if xmin == None else min(xmin, mx.min())
    xmax = mx.max() if xmax == None else max(xmax, mx.max())
    ymax = my.max() if ymax == None else max(ymax, my.max())

  if sensor:
    sx = np.array(sensor['x'], dtype=np.float64)
    sy = np.array(sensor['y'], dtype=np.float64)

    xmin = sx.min() if xmin == None else min(xmin, sx.min())
    xmax = sx.max() if xmax == None else max(xmax, sx.max())
    ymax = sy.max() if ymax == None else max(ymax, sy.max())

  while n_bounces >= 0:
    bestR, bestX, bestY = None, None, None
    bestType = None
    bestTangent = None
    for mx, my in zip(mxs, mys):
      x, y, dx, dy = findSegments(x0, y0, vx, vy, mx, my)
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY = r[irmin], x[irmin], y[irmin]
          bestType = 'mirror'
          bestTangent = [dx[irmin], dy[irmin]]
    if sensor:
      x, y, dx, dy = findSegments(x0, y0, vx, vy, sx, sy)
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY = r[irmin], x[irmin], y[irmin]
          bestType = 'on sensor'
    ## Add a virtual layer to pick up rays escaping backwards
    if bestType == None:
      x, y, dx, dy = findSegments(x0, y0, vx, vy,
                                  np.array([xmin, xmax], dtype=np.float64),
                                  np.array([ymax, ymax], dtype=np.float64))
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, x0, y0 = r[irmin], x[irmin], y[irmin]
          bestType = 'bounced back'
    else:
      x0, y0 = bestX, bestY
    if bestType == None:
      x, y, dx, dy = findSegments(x0, y0, vx, vy,
                                  np.array([xmin, xmax], dtype=np.float64),
                                  np.array([0, 0], dtype=np.float64))
      if len(x) > 0:
        r = getDist(x0, y0, x, y, vx, vy)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
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

if __name__ == '__main__':
  #plt.style.use('ROOT')
  hep.style.use(hep.style.CMS)

  par_height = 1200
  par_width = 1200
  par_din = 1200
  par_dout = 460
  #par_angle = 80
  par_angle = 20
  par_sensor_curv = 325 ## sensor curvature (325mm for R12860)

  par_n_rays = 101
  inc_angle = -30

  #config = make_planar(par_din, par_dout, par_angle, par_width, par_height)
  config = make_winston(par_din, par_dout, par_angle, par_width, par_height)
  config['sensor'] = make_sensor(par_dout, sensor_curv=par_sensor_curv)
  # print(config)

  rmax = 0
  for shape in config['mirrors']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    plt.plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw sensor surface
  x, y = config['sensor']['x'], config['sensor']['y']
  plt.plot(x, y, 'g', linewidth=2)

  ## Draw axis
  plt.plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
  plt.plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

  ## Trace a few sample rays
  for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
    xs, ys, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensor=config['sensor'])
    color = {'exit':'b', 'bounced back':'r', 'bounce limit':'r', 'on sensor':'g'}
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.plot(xs, ys, color[exit_type]+'-', linewidth=0.5)

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2 * rmax, 1.2 * rmax)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

  ## Start scanning over the incident angle
  inc_angles = np.linspace(0, 90, 200)
  frac_pass = np.zeros(len(inc_angles))
  frac_entr = np.zeros(len(inc_angles))
  for i, inc_angle in enumerate(tqdm(inc_angles)):
    n_pass, n_entr = 0, 0
    for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
      _, _, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensor=config['sensor'])
      if exit_type == 'on sensor':
        n_pass += 1
      elif exit_type == 'bounced back':
        n_entr += 1
    frac_pass[i] = n_pass/par_n_rays
    frac_entr[i] = n_entr/par_n_rays

  plt.plot(inc_angles, frac_pass, 'b.-', label='on sensor')
  plt.plot(inc_angles, frac_entr, 'r.-', label='bounced back')
  plt.xlabel('Indicent angle (deg)')
  plt.ylabel('Fraction')
  plt.legend()
  plt.show()
