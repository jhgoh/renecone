#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm

import sys
sys.path.append('python')
from ConeProfile import *

def _build_segments(mirrors, label=None):
  """Convert polylines into a list of labeled line segments."""
  segments = []
  for shape in mirrors:
    x = np.asarray(shape['x'])
    y = np.asarray(shape['y'])
    for p1, p2 in zip(np.column_stack((x[:-1], y[:-1])),
                      np.column_stack((x[1:], y[1:]))):
      segments.append((p1, p2, label))
  return segments


def _ray_segment_intersection(pos, vel, p1, p2):
  """Return distance along ray and intersection point with a segment."""
  r = vel
  s = p2 - p1
  denom = r[0] * s[1] - r[1] * s[0]
  if np.isclose(denom, 0):
    return None
  t = ((p1[0] - pos[0]) * s[1] - (p1[1] - pos[1]) * s[0]) / denom
  u = ((p1[0] - pos[0]) * r[1] - (p1[1] - pos[1]) * r[0]) / denom
  if t > 1e-9 and 0 <= u <= 1:
    return t, p1 + u * s
  return None


def propagate(x, y, angle, mirrors, sensors=None, n_bounces=20):
  """Propagate a ray through a set of mirror segments and a sensor surface.

  Parameters
  ----------
  x, y : float
      Starting coordinates of the ray.
  angle : float
      Direction of the ray in degrees, measured counterclockwise from the incidence
  mirrors : sequence
      Collection of mirror mirrors to intersect with.
  sensors : dict, optional
      Shape describing the sensor surface to treat as an exit.
  n_bounces : int, optional
      Stop after this many reflections if the ray has not exited.

  Returns
  -------
  xs, ys : ndarray
      Coordinates of the ray path.
  exit_type : {'exit', 'bounced back', 'bounce limit', 'on sensor'}
      Reason the propagation terminated.
  """
  pos = np.array([x, y], dtype=float)
  rad = np.deg2rad(angle-90)
  vel = np.array([np.cos(rad), np.sin(rad)], dtype=float)

  xs = [pos[0]]
  ys = [pos[1]]
  segments = _build_segments(mirrors, 'mirror')
  if sensors is not None:
    segments.extend(_build_segments([sensors], 'sensors'))
  height = max(np.max(s['y']) for s in mirrors)
  bounces = 0

  while True:
    best = None
    best_seg = None
    best_label = None
    for p1, p2, label in segments:
      res = _ray_segment_intersection(pos, vel, p1, p2)
      if res is None:
        continue
      t, ipt = res
      if best is None or t < best[0]:
        best = (t, ipt)
        best_seg = (p1, p2)
        best_label = label

    if best is None:
      if vel[1] < 0:  # extend ray to exit plane y=0
        t = -pos[1] / vel[1]
        xs.append(pos[0] + t * vel[0])
        ys.append(0.0)
        exit_type = 'exit'
      else:  # ray escapes back out the bounced back
        t = (height - pos[1]) / vel[1]
        xs.append(pos[0] + t * vel[0])
        ys.append(height)
        exit_type = 'bounced back'
      break

    pos = best[1]
    xs.append(pos[0])
    ys.append(pos[1])

    if best_label == 'sensors':
      exit_type = 'on sensor'
      break
    if pos[1] <= 0:
      exit_type = 'exit'
      break
    if pos[1] >= height:
      exit_type = 'bounced back'
      break

    seg_vec = best_seg[1] - best_seg[0]
    normal = np.array([-seg_vec[1], seg_vec[0]])
    normal /= np.linalg.norm(normal)
    vel = vel - 2 * np.dot(vel, normal) * normal

    bounces += 1
    if bounces >= n_bounces:
      exit_type = 'bounce limit'
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
  inc_angle = 10

  #config = make_planar(par_din, par_dout, par_angle, par_width, par_height)
  config = make_winston(par_din, par_dout, par_angle, par_width, par_height)
  config['sensors'] = make_sensors(par_dout, sensor_curv=par_sensor_curv)
  # print(config)

  rmax = 0
  for shape in config['mirrors']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    plt.plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw sensor surface
  x, y = config['sensors']['x'], config['sensors']['y']
  plt.plot(x, y, 'g', linewidth=2)

  ## Draw axis
  plt.plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
  plt.plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

  ## Trace a few sample rays
  for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
    xs, ys, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensors=config['sensors'])
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
      _, _, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensors=config['sensors'])
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
