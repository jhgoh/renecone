#!/usr/bin/env python
import numpy as np
from typing import Dict, Optional, Sequence

def make_planar(din: float, dout: float, angle: float, width: float, height: float,
                sides: Optional[Sequence[str]] = ['left', 'right']) -> Dict[str, object]:
  """Construct a planar (flat mirror) concentrator description."""
  config: Dict[str, object] = {
    'name': 'planar',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'angle': angle,
    'mirrors': [],
  }

  ## Left and right walls
  if 'left' in sides:
    config['mirrors'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  if 'right' in sides:
    config['mirrors'].append({'x': [width / 2, width / 2], 'y': [0, height]})

  ## Main mirrors
  h = dout * np.tan(np.deg2rad(90 - angle))
  if 'left' in sides:
    config['mirrors'].append({'x': [-dout / 2, -din / 2], 'y': [0, h]})
  if 'right' in sides:
    config['mirrors'].append({'x': [dout / 2, din / 2], 'y': [0, h]})

  return config


def make_winston(din: float, dout: float, crit_angle: float,
                 width: float, height: float,
                 n_points: int = 512,
                 sides: Optional[Sequence[str]] = ['left', 'right']) -> Dict[str, object]:
  """Generate a Winston cone profile sampled along the requested sides."""
  config: Dict[str, object] = {
    'name': 'winston',
    'n_walls': 4,
    'din': din,
    'dout': dout,
    'crit_angle': crit_angle,
    'mirrors': [],
  }

  ## Left and right walls
  if 'left' in sides:
    config['mirrors'].append({'x': [-width / 2, -width / 2], 'y': [0, height]})
  if 'right' in sides:
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
  y_vals = x_vals**2 / (4 * focal_length)

  ## Rotate & translate the mirror
  x_coords = x_vals * cos_c - (y_vals - focal_length) * sin_c - r_out
  y_coords = x_vals * sin_c + (y_vals - focal_length) * cos_c

  ## Clip points outside of the main volume
  sel_idx = (np.abs(x_coords) <= din / 2) & (y_coords >= 0) & (y_coords <= height)
  x_coords = x_coords[sel_idx]
  y_coords = y_coords[sel_idx]

  if 'left' in sides:
    config['mirrors'].append({'x': -x_coords, 'y': y_coords})
  if 'right' in sides:
    config['mirrors'].append({'x': x_coords, 'y': y_coords})

  return config


def make_sensor(dout: float, sensor_curv: Optional[float] = None,
                n_points: int = 25,
                sides: Optional[Sequence[str]] = ['left', 'right']) -> Dict[str, Sequence[float]]:
  """Generate a sensing surface at the exit plane."""
  xmin = -dout / 2 if 'left' in sides else 0
  xmax = dout / 2 if 'right' in sides else 0

  if sensor_curv is not None:
    x = np.linspace(xmin, xmax, n_points)
    sagitta = np.sqrt(sensor_curv**2 - (dout / 2)**2)
    y = np.sqrt(sensor_curv**2 - x**2) - sagitta
    return {'x': x, 'y': y}
  return {'x': [xmin, xmax], 'y': [0, 0]}

