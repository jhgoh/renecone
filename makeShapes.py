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

def propagate(x, y, vx, vy, mirrors, nstep=10000):
  for i in range(nstep):
    pass


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

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2 * rmax, 1.2 * rmax)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

