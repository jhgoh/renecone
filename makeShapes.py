#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def make_planar(din, dout, angle, width=1200, height=1200):
  result = {
    'name': 'planar',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'angle': angle,
    'shapes': [],
  }

  ## Left and right walls
  result['shapes'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  result['shapes'].append({'x': [width / 2, width / 2], 'y': [0, height]})

  ## Main mirrors
  h = dout * np.tan(np.deg2rad(angle))
  result['shapes'].append({'x': [dout / 2, din / 2], 'y': [0, h]})
  result['shapes'].append({'x': [-dout / 2, -din / 2], 'y': [0, h]})

  return result


def make_winston(din, dout, crit_angle, width=1200, height=1200, n_points=25):
  result = {
    'name': 'winston',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'crit_angle': crit_angle,
    'shapes': [],
  }

  ## Left and right walls
  result['shapes'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  result['shapes'].append({'x': [width / 2, width / 2], 'y': [0, height]})

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

  result['shapes'].append({'x': x_coords, 'y': y_coords})
  result['shapes'].append({'x': -x_coords, 'y': y_coords})

  return result

def _build_segments(mirrors):
  """Convert mirror polylines into a list of line segments."""
  segments = []
  for shape in mirrors:
    x = np.asarray(shape['x'])
    y = np.asarray(shape['y'])
    for p1, p2 in zip(np.column_stack((x[:-1], y[:-1])),
                      np.column_stack((x[1:], y[1:]))):
      segments.append((p1, p2))
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


def propagate(x, y, angle, mirrors, n_bounces=20):
  """Propagate a ray through a set of mirror segments.

  Parameters
  ----------
  x, y : float
      Starting coordinates of the ray.
  angle : float
      Direction of the ray in degrees, measured counterclockwise from the
      positive x-axis.
  mirrors : sequence
      Collection of mirror shapes to intersect with.
  n_bounces : int, optional
      Maximum number of reflections to calculate.
  """
  pos = np.array([x, y], dtype=float)
  rad = np.deg2rad(angle)
  vel = np.array([np.cos(rad), np.sin(rad)], dtype=float)

  xs = [pos[0]]
  ys = [pos[1]]
  segments = _build_segments(mirrors)

  for _ in range(n_bounces):
    best = None
    best_seg = None
    for p1, p2 in segments:
      res = _ray_segment_intersection(pos, vel, p1, p2)
      if res is None:
        continue
      t, ipt = res
      if best is None or t < best[0]:
        best = (t, ipt)
        best_seg = (p1, p2)

    if best is None:
      if vel[1] < 0:  # extend ray to exit plane y=0
        t = -pos[1] / vel[1]
        xs.append(pos[0] + t * vel[0])
        ys.append(0.0)
      break

    pos = best[1]
    xs.append(pos[0])
    ys.append(pos[1])

    if pos[1] <= 0:
      break

    seg_vec = best_seg[1] - best_seg[0]
    normal = np.array([-seg_vec[1], seg_vec[0]])
    normal /= np.linalg.norm(normal)
    vel = vel - 2 * np.dot(vel, normal) * normal

  return np.array(xs), np.array(ys)


if __name__ == '__main__':
  par_din = 1200
  par_dout = 500
  par_angle = 25
  # par_angle = 10

  # result = make_planar(par_din, par_dout, par_angle)
  result = make_winston(par_din, par_dout, par_angle)
  # print(result)

  rmax = 0
  for shape in result['shapes']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    plt.plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw axis
  plt.plot([-result['din'] / 2, result['din'] / 2], [0, 0], '-.k', linewidth=0.5)
  plt.plot([0, 0], [0, 1.5 * rmax], '-.k', linewidth=0.5)

  ## Trace a few sample rays
  height = max(np.max(s['y']) for s in result['shapes'])
  for x0 in np.linspace(-result['din'] / 2 * 0.9, result['din'] / 2 * 0.9, 5):
    xs, ys = propagate(x0, height - 1, -90, result['shapes'])
    plt.plot(xs, ys, 'r-')

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2 * rmax, 1.2 * rmax)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

