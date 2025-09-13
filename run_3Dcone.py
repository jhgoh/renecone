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

tol: float = 1e-7 ## Numerical tolerance

@njit(cache=True)
def findSegments(x0: float, y0: float, z0: float,
                 vx: float, vy: float, vz: float,
                 mw: np.ndarray, mh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray]:
  dmw = mw[1:] - mw[:-1]
  dmh = mh[1:] - mh[:-1]

  vw = np.hypot(vx, vz)

  dmh2 = dmh**2
  dmw2 = dmw**2

  a = dmh2*vw*vw - dmw2*vz*vz
  b = dmh2*(vx*x0 + vz*z0) - dmw*vy*((y0-mh[:-1])*dmw + mw[:-1]*dmh)
  c = dmh2*(x0*x0 + z0*z0) - ((y0-mh[:-1])*dmw + mw[:-1]*dmh)**2

  t = np.full(dmw.shape, np.inf)
  
  ## a=0, b!=0 case: 1 polynomial
  is2pol = np.abs(a) > tol
  mask = (~is2pol) & (np.abs(b) > tol)
  t[mask] = c[mask]/2/b[mask]

  ## a!=0, det=0 case: 2nd order polynomial but 1 solution
  det = b**2 - a*c
  has2 = np.abs(det) > tol
  mask = is2pol & ~has2
  t[mask] = -b[mask]/a[mask]

  ## det > 0 case: two solutions. We take smaller one with positive solution
  sol1 = np.full(t.shape, np.inf)
  sol2 = np.full(t.shape, np.inf)
  mask = is2pol & has2
  sol1[mask] = (-b[mask] + np.sqrt(det[mask]))/a[mask]
  sol2[mask] = (-b[mask] - np.sqrt(det[mask]))/a[mask]
  sol1[sol1 < tol] = np.inf
  sol2[sol2 < tol] = np.inf
  t[mask] = np.minimum(sol1[mask], sol2[mask])

  x = x0 + t * vx
  y = y0 + t * vy
  z = z0 + t * vz
  w = np.hypot(x,z)

  hit = (t > tol) & ((y - mh[1:])*(y - mh[:-1]) < 0) & ((w - mw[1:])*(w - mw[:-1]) < 0)

  ny = -dmw
  nw = dmh
  nx = nw*x/w # = nw * cos(phi)
  nz = nw*z/w # = nw * sin(phi)

  return x[hit], y[hit], z[hit], nx[hit], ny[hit], nz[hit]

@njit(cache=True)
def getDist(x0:float, y0:float, z0:float,
            x:np.ndarray, y:np.ndarray, z:np.ndarray,
            vx:float, vy:float, vz:float) -> float:
  dx, dy, dz = x - x0, y - y0, z - z0
  vr = (vx**2 + vy**2 + vz**2)**0.5
  r = (vx * dx + vy * dy + vz * dz) / vr

  return r

@njit(cache=True)
def reflect(vx:float, vy:float, vz:float,
            nx:float, ny:float, nz:float) -> Tuple[float, float, float]:
  nr = (nx**2 + ny**2 + nz**2)**0.5
  nx, ny, nz = nx/nr, ny/nr, nz/nr

  norm = vx * nx + vy * ny + vz * nz
  if norm > 0:
    nx, ny, nz = -nx, -ny, -nz
    norm = -norm

  ux = vx - 2 * norm * nx
  uy = vy - 2 * norm * ny
  uz = vz - 2 * norm * nz

  return ux, uy, uz  

def propagate(x0: float, y0: float, z0: float, angle: float,
              mirrors: List[Dict[str, Sequence[float]]],
              sensor: Optional[Dict[str, Sequence[float]]] = None,
              n_bounces: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
  rad = np.deg2rad(angle-90) ## initial light ray of theta=0 points downwards
  vx, vy, vz = np.cos(rad), np.sin(rad), 0 ## set the initial direction of light ray. we don't consider z-component (through the screen)
  xs, ys, zs = [x0], [y0], [z0]
  exit_type = 'bounce limit'

  rmin, rmax, ymax = None, None, None

  mws, mhs = [], []
  for mirror in mirrors:
    mw = np.array(mirror['x'], dtype=np.float64)
    mh = np.array(mirror['y'], dtype=np.float64)
    mws.append(mw)
    mhs.append(mh)

    rmax = mw.max() if rmax == None else max(rmax, np.abs(mw.min()), mw.max())
    ymax = mh.max() if ymax == None else max(ymax, mh.max())

  if sensor:
    sx = np.array(sensor['x'], dtype=np.float64)
    sy = np.array(sensor['y'], dtype=np.float64)

    rmax = sx.max() if rmax == None else max(rmax, np.abs(sx.min()), sx.max())
    ymax = sy.max() if ymax == None else max(ymax, sy.max())

  while n_bounces >= 0:
    bestR, bestX, bestY, bestZ = None, None, None, None
    bestType = None
    bestTangent = None
    for mw, mh in zip(mws, mhs):
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, mw, mh)
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'mirror'
          bestTangent = [nx[irmin], ny[irmin], nz[irmin]]
    if sensor:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz, sx, sy)
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'on sensor'
    ## Add a virtual layer to pick up rays escaping backwards
    if bestType == None:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([-rmax, rmax]), np.array([ymax, ymax]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, x0, y0, z0 = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'bounced back'
    else:
      x0, y0, z0 = bestX, bestY, bestZ
    if bestType == None:
      x, y, z, nx, ny, nz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([-rmax, rmax]), np.array([0, 0]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, x, y, z, vx, vy, vz)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, x0, y0, z0 = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'exit'

    xs.append(x0)
    ys.append(y0)
    zs.append(z0)

    if bestType == 'mirror':
      n_bounces -= 1
      vx, vy, vz = reflect(vx, vy, vz, *bestTangent)
    else:
      exit_type = bestType if bestType else 'bounced back'
      break

  return np.array(xs), np.array(ys), np.array(zs), exit_type

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

  par_n_rays = 51
  inc_angle = -30

  #config = make_planar(par_din, par_dout, par_angle, par_width, par_height)
  config = make_winston(par_din, par_dout, par_angle, par_width, par_height)
  config['sensor'] = make_sensor(par_dout, sensor_curv=par_sensor_curv)
  # print(config)

  fig, axes = plt.subplots(1, 2, figsize=(15,7))

  rmax = 0
  for shape in config['mirrors']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    axes[0].plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw sensor surface
  x, y = config['sensor']['x'], config['sensor']['y']
  axes[0].plot(x, y, 'g', linewidth=2)

  ## Draw axis
  axes[0].plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
  axes[0].plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

  ## Trace a few sample rays
  for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
    z0 = 0
    xs, ys, zs, exit_type = propagate(x0, par_height - 1, z0, inc_angle, config['mirrors'], sensor=config['sensor'])
    color = {'exit':'b', 'bounced back':'r', 'bounce limit':'r', 'on sensor':'g'}
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].plot(xs, np.hypot(ys, zs), color[exit_type]+'-', linewidth=0.5)

    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('z (mm)')
    axes[1].plot(ys, zs, color[exit_type]+'-', linewidth=0.5)

  axes[0].set_xlim(-rmax, rmax)
  axes[0].set_ylim(-0.2 * rmax, 1.2 * rmax)
  axes[0].set_aspect('equal', adjustable='box')

  axes[1].set_xlim(-rmax, rmax)
  axes[1].set_ylim(-rmax, rmax)
  axes[1].set_aspect('equal', adjustable='box')
  plt.show()

  ### Start scanning over the incident angle
  #inc_angles = np.linspace(0, 90, 200)
  #frac_pass = np.zeros(len(inc_angles))
  #frac_entr = np.zeros(len(inc_angles))
  #for i, inc_angle in enumerate(tqdm(inc_angles)):
  #  n_pass, n_entr = 0, 0
  #  for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
  #    _, _, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensor=config['sensor'])
  #    if exit_type == 'on sensor':
  #      n_pass += 1
  #    elif exit_type == 'bounced back':
  #      n_entr += 1
  #  frac_pass[i] = n_pass/par_n_rays
  #  frac_entr[i] = n_entr/par_n_rays

  #plt.plot(inc_angles, frac_pass, 'b.-', label='on sensor')
  #plt.plot(inc_angles, frac_entr, 'r.-', label='bounced back')
  #plt.xlabel('Indicent angle (deg)')
  #plt.ylabel('Fraction')
  #plt.legend()
  #plt.show()
