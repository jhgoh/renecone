#!/usr/bin/env python
import numpy as np

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


def make_sensors(dout, sensor_curv=None, n_points=25):
  """Generate a sensing surface at the exit plane."""
  if sensor_curv is not None:
    x = np.linspace(-dout / 2, dout / 2, n_points)
    sagitta = np.sqrt(sensor_curv**2 - (dout / 2)**2)
    y = np.sqrt(sensor_curv**2 - x**2) - sagitta
    return {'x': x, 'y': y}
  return {'x': [-dout / 2, dout / 2], 'y': [0, 0]}

