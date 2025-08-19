#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_planar(din, dout, angle, width, height):
  config = {
    'name': 'planar',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'angle': angle,
    'mirrors': [],
  }

  ## Left and right walls
  config['mirrors'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  config['mirrors'].append({'x': [width / 2, width / 2], 'y': [0, height]})

  ## Main mirrors
  h = dout * np.tan(np.deg2rad(90-angle))
  config['mirrors'].append({'x': [dout / 2, din / 2], 'y': [0, h]})
  config['mirrors'].append({'x': [-dout / 2, -din / 2], 'y': [0, h]})

  return config


def make_winston(din, dout, crit_angle, width, height, n_points=512):
  config = {
    'name': 'winston',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'crit_angle': crit_angle,
    'mirrors': [],
  }

  ## Left and right walls
  config['mirrors'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  config['mirrors'].append({'x': [width / 2, width / 2], 'y': [0, height]})

  ## Main mirrors
  crit_angle = np.deg2rad(crit_angle)
  r_in = din / 2
  r_out = dout / 2
  sin_c, cos_c = np.sin(crit_angle), np.cos(crit_angle)
  intrinsic_height = (r_in + r_out) / np.tan(crit_angle)
  focal_length = r_out * (1 + sin_c)

  ## Build parabola in mirror axis
  d_1 = 2 * r_out * cos_c
  d_2 = intrinsic_height * np.sin(2 * crit_angle) / cos_c
  x_vals = np.linspace(d_1, d_2, n_points)
  y_vals = x_vals ** 2 / 4 / focal_length

  ## Rotate & translate the mirror
  x_coords = x_vals * cos_c - (y_vals - focal_length) * sin_c - r_out
  y_coords = x_vals * sin_c + (y_vals - focal_length) * cos_c

  ## Clip points outside of the main volume
  sel_idx = (np.abs(x_coords) <= din/2) & (y_coords >= 0) & (y_coords <= height)
  x_coords = x_coords[sel_idx]
  y_coords = y_coords[sel_idx]

  config['mirrors'].append({'x': x_coords, 'y': y_coords})
  config['mirrors'].append({'x': -x_coords, 'y': y_coords})

  return config


def make_pmt(dout, pmt_curv=None, n_points=25):
  """Generate a PMT surface at the exit plane."""
  if pmt_curv is not None:
    x = np.linspace(-dout / 2, dout / 2, n_points)
    sagitta = np.sqrt(pmt_curv**2 - (dout / 2)**2)
    y = np.sqrt(pmt_curv**2 - x**2) - sagitta
    return {'x': x, 'y': y}
  return {'x': [-dout / 2, dout / 2], 'y': [0, 0]}

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


def propagate(x, y, angle, mirrors, pmt=None, n_bounces=20):
  """Propagate a ray through a set of mirror segments and a PMT surface.

  Parameters
  ----------
  x, y : float
      Starting coordinates of the ray.
  angle : float
      Direction of the ray in degrees, measured counterclockwise from the incidence
  mirrors : sequence
      Collection of mirror mirrors to intersect with.
  pmt : dict, optional
      Shape describing the PMT surface to treat as an exit.
  n_bounces : int, optional
      Stop after this many reflections if the ray has not exited.

  Returns
  -------
  xs, ys : ndarray
      Coordinates of the ray path.
  exit_type : {'exit', 'entrance', 'bounce_limit', 'pmt'}
      Reason the propagation terminated.
  """
  pos = np.array([x, y], dtype=float)
  rad = np.deg2rad(angle-90)
  vel = np.array([np.cos(rad), np.sin(rad)], dtype=float)

  xs = [pos[0]]
  ys = [pos[1]]
  segments = _build_segments(mirrors, 'mirror')
  if pmt is not None:
    segments.extend(_build_segments([pmt], 'pmt'))
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
      else:  # ray escapes back out the entrance
        t = (height - pos[1]) / vel[1]
        xs.append(pos[0] + t * vel[0])
        ys.append(height)
        exit_type = 'entrance'
      break

    pos = best[1]
    xs.append(pos[0])
    ys.append(pos[1])

    if best_label == 'pmt':
      exit_type = 'pmt'
      break
    if pos[1] <= 0:
      exit_type = 'exit'
      break
    if pos[1] >= height:
      exit_type = 'entrance'
      break

    seg_vec = best_seg[1] - best_seg[0]
    normal = np.array([-seg_vec[1], seg_vec[0]])
    normal /= np.linalg.norm(normal)
    vel = vel - 2 * np.dot(vel, normal) * normal

    bounces += 1
    if bounces >= n_bounces:
      exit_type = 'bounce_limit'
      break

  return np.array(xs), np.array(ys), exit_type


if __name__ == '__main__':
  par_height = 1200
  par_width = 1200
  par_din = 1200
  par_dout = 460
  #par_angle = 45
  par_angle = 80
  par_pmt_curv = 325 ## PMT curvature (325mm for R12860)

  par_n_rays = 101
  inc_angle = 10

  config = make_planar(par_din, par_dout, par_angle, par_width, par_height)
  #config = make_winston(par_din, par_dout, par_angle, par_width, par_height)
  config['pmt'] = make_pmt(par_dout, pmt_curv=par_pmt_curv)
  # print(config)

  rmax = 0
  for shape in config['mirrors']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    plt.plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw PMT surface
  x, y = config['pmt']['x'], config['pmt']['y']
  plt.plot(x, y, 'g', linewidth=2)

  ## Draw axis
  plt.plot([-config['din'] / 2, config['din'] / 2], [0, 0], '-.k', linewidth=0.5)
  plt.plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

  ## Trace a few sample rays
  for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
    xs, ys, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], pmt=config['pmt'])
    color = {'exit':'b', 'entrance':'r', 'bounce_limit':'r', 'pmt':'g'}
    plt.plot(xs, ys, color[exit_type]+'-', linewidth=0.5)

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2 * rmax, 1.2 * rmax)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

  ## Start scanning over the incident angle
  inc_angles = np.linspace(0, 90, 100)
  frac_pass = np.zeros(len(inc_angles))
  frac_entr = np.zeros(len(inc_angles))
  for i, inc_angle in enumerate(tqdm(inc_angles)):
    n_pass, n_entr = 0, 0
    for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
      _, _, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], pmt=config['pmt'])
      if exit_type == 'pmt':
        n_pass += 1
      elif exit_type == 'entrance':
        n_entr += 1
    frac_pass[i] = n_pass/par_n_rays
    frac_entr[i] = n_entr/par_n_rays

  plt.plot(inc_angles, frac_pass, 'b.-')
  plt.plot(inc_angles, frac_entr, 'r.-')
  plt.show()
