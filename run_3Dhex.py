#!/usr/bin/env python
import argparse
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from tqdm import tqdm


tol: float = 1e-7


def _parse_csv_floats(value: str) -> np.ndarray:
  return np.array([float(v.strip()) for v in value.split(',') if v.strip() != ''], dtype=np.float64)


def _regular_polygon(radius: float, n_sides: int = 6) -> np.ndarray:
  phases = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
  return np.column_stack([radius * np.cos(phases), radius * np.sin(phases)])


def _point_in_convex_polygon(px: float, pz: float, poly: np.ndarray) -> bool:
  signs = []
  n = len(poly)
  for i in range(n):
    x1, z1 = poly[i]
    x2, z2 = poly[(i + 1) % n]
    cross = (x2 - x1) * (pz - z1) - (z2 - z1) * (px - x1)
    if abs(cross) > tol:
      signs.append(np.sign(cross))
  if len(signs) == 0:
    return False
  signs = np.array(signs)
  return np.all(signs >= 0) or np.all(signs <= 0)


def _point_in_triangle(point: np.ndarray, tri: np.ndarray) -> bool:
  a, b, c = tri
  v0 = c - a
  v1 = b - a
  v2 = point - a

  dot00 = np.dot(v0, v0)
  dot01 = np.dot(v0, v1)
  dot02 = np.dot(v0, v2)
  dot11 = np.dot(v1, v1)
  dot12 = np.dot(v1, v2)
  denom = dot00 * dot11 - dot01 * dot01
  if abs(denom) < tol:
    return False

  u = (dot11 * dot02 - dot01 * dot12) / denom
  v = (dot00 * dot12 - dot01 * dot02) / denom
  return (u >= -tol) and (v >= -tol) and (u + v <= 1 + tol)


def _point_in_quad(point: np.ndarray, quad: np.ndarray) -> bool:
  return _point_in_triangle(point, quad[[0, 1, 2]]) or _point_in_triangle(point, quad[[0, 2, 3]])


def _reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
  n = n / np.linalg.norm(n)
  proj = np.dot(v, n)
  if proj > 0:
    n = -n
    proj = -proj
  return v - 2 * proj * n


def make_faceted_hex_geometry(
    din: float,
    dout: float,
    height: float,
    y_knots: np.ndarray,
    scale_knots: np.ndarray,
) -> Dict[str, np.ndarray]:
  """Create hex frustum from stacked rings (multi-panel side walls)."""
  if len(y_knots) != len(scale_knots):
    raise ValueError('`y-knots` and `scale-knots` must have same length')
  if np.any(y_knots < 0) or np.any(y_knots > 1):
    raise ValueError('`y-knots` must be in [0, 1]')

  order = np.argsort(y_knots)
  y_sorted = y_knots[order]
  s_sorted = scale_knots[order]

  keep_mid = (y_sorted > 0) & (y_sorted < 1)
  y_prof = np.concatenate(([0.0], y_sorted[keep_mid], [1.0]))
  s_prof = np.concatenate(([1.0], s_sorted[keep_mid], [1.0]))

  r_out = dout / 2
  r_in = din / 2

  rings: List[np.ndarray] = []
  for frac, scale in zip(y_prof, s_prof):
    base = r_out + (r_in - r_out) * frac
    radius = max(base * scale, 1.0)
    xz = _regular_polygon(radius, n_sides=6)
    ring = np.column_stack([xz[:, 0], np.full(6, frac * height), xz[:, 1]])
    rings.append(ring)

  faces = []
  normals = []
  center = np.array([0.0, height / 2, 0.0])
  for ridx in range(len(rings) - 1):
    upper = rings[ridx + 1]
    lower = rings[ridx]
    for i in range(6):
      ip1 = (i + 1) % 6
      quad = np.array([upper[i], upper[ip1], lower[ip1], lower[i]], dtype=np.float64)
      n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
      n_norm = np.linalg.norm(n)
      if n_norm < tol:
        continue
      n = n / n_norm
      face_center = quad.mean(axis=0)
      if np.dot(n, center - face_center) < 0:
        n = -n
      faces.append(quad)
      normals.append(n)

  return {
    'rings': np.array(rings),
    'faces': np.array(faces),
    'normals': np.array(normals),
    'top_xz': rings[-1][:, [0, 2]],
    'bot_xz': rings[0][:, [0, 2]],
    'top_y': rings[-1][0, 1],
    'bot_y': rings[0][0, 1],
  }


def propagate_hex(
    x0: float,
    y0: float,
    z0: float,
    angle: float,
    geom: Dict[str, np.ndarray],
    n_bounces: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
  rad = np.deg2rad(angle - 90)
  p = np.array([x0, y0, z0], dtype=np.float64)
  v = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float64)
  xs, ys, zs = [x0], [y0], [z0]

  for _ in range(n_bounces):
    best_t = np.inf
    best_type = None
    best_point = None
    best_normal = None

    for face, normal in zip(geom['faces'], geom['normals']):
      denom = np.dot(v, normal)
      if abs(denom) <= tol:
        continue
      t = np.dot(face[0] - p, normal) / denom
      if t <= tol or t >= best_t:
        continue
      hit = p + t * v
      if _point_in_quad(hit, face):
        best_t = t
        best_type = 'mirror'
        best_point = hit
        best_normal = normal

    if abs(v[1]) > tol:
      t_top = (geom['top_y'] - p[1]) / v[1]
      if t_top > tol and t_top < best_t:
        hit = p + t_top * v
        if _point_in_convex_polygon(hit[0], hit[2], geom['top_xz']):
          best_t, best_type, best_point = t_top, 'bounced back', hit

      t_bot = (geom['bot_y'] - p[1]) / v[1]
      if t_bot > tol and t_bot < best_t:
        hit = p + t_bot * v
        if _point_in_convex_polygon(hit[0], hit[2], geom['bot_xz']):
          best_t, best_type, best_point = t_bot, 'on sensor', hit
        else:
          best_t, best_type, best_point = t_bot, 'exit', hit

    if best_type is None:
      return np.array(xs), np.array(ys), np.array(zs), 'exit'

    p = best_point
    xs.append(p[0])
    ys.append(p[1])
    zs.append(p[2])

    if best_type != 'mirror':
      return np.array(xs), np.array(ys), np.array(zs), best_type

    v = _reflect(v, best_normal)

  return np.array(xs), np.array(ys), np.array(zs), 'bounce limit'


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Faceted hexagonal cone ray tracing scan')
  parser.add_argument('-o', '--output', required=True,
                      help='Path to the CSV file where scan results are stored')
  parser.add_argument('-q', '--quiet', action='store_true',
                      help='Silent mode: skip drawing figures')
  parser.add_argument('--height', type=float, default=1200, help='Cone height (mm)')
  parser.add_argument('--din', type=float, default=1200, help='Hex entrance circum-diameter (mm)')
  parser.add_argument('--dout', type=float, default=460, help='Hex exit circum-diameter (mm)')
  parser.add_argument('--n-rays-vis', type=int, default=51, help='Rays in preview figure')
  parser.add_argument('--n-rays', type=int, default=10000, help='Rays per scan point')
  parser.add_argument('--inc-angle', type=float, default=20, help='Incident angle for preview (deg)')
  parser.add_argument('--scan-min', type=float, default=0, help='Minimum incident angle for scan (deg)')
  parser.add_argument('--scan-max', type=float, default=90, help='Maximum incident angle for scan (deg)')
  parser.add_argument('--scan-steps', type=int, default=200, help='Number of scan points')
  parser.add_argument('--y-knots', type=str, default='0.25,0.5,0.75',
                      help='Comma-separated knot positions along height [0..1] to place intermediate panels')
  parser.add_argument('--scale-knots', type=str, default='1.0,1.0,1.0',
                      help='Comma-separated radius multipliers at y-knots for panel-shape tuning')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  hep.style.use(hep.style.CMS)

  y_knots = _parse_csv_floats(args.y_knots)
  scale_knots = _parse_csv_floats(args.scale_knots)
  if len(y_knots) != len(scale_knots):
    raise ValueError('`--y-knots` and `--scale-knots` lengths must match')

  geom = make_faceted_hex_geometry(args.din, args.dout, args.height, y_knots, scale_knots)

  if not args.quiet:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plt.tight_layout(pad=2.0)

    rings = geom['rings']
    for ring in rings:
      loop = np.vstack([ring, ring[0]])
      axes[0].plot(loop[:, 0], loop[:, 1], 'k', alpha=0.5)
    for i in range(6):
      meridian_x = [ring[i, 0] for ring in rings]
      meridian_y = [ring[i, 1] for ring in rings]
      axes[0].plot(meridian_x, meridian_y, 'k')

    top_loop = np.vstack([geom['top_xz'], geom['top_xz'][0]])
    bot_loop = np.vstack([geom['bot_xz'], geom['bot_xz'][0]])
    axes[1].plot(top_loop[:, 0], top_loop[:, 1], 'k')
    axes[1].plot(bot_loop[:, 0], bot_loop[:, 1], 'g', linewidth=2)

    rmax = np.max(np.abs(geom['top_xz']))
    span = rmax * 0.9
    for x0 in np.linspace(-span, span, args.n_rays_vis):
      z0 = 0.0
      if not _point_in_convex_polygon(x0, z0, geom['top_xz']):
        continue
      xs, ys, zs, exit_type = propagate_hex(x0, args.height - 1, z0, args.inc_angle, geom)
      color = {'exit': 'b', 'bounced back': 'r', 'bounce limit': 'r', 'on sensor': 'g'}
      axes[0].plot(xs, ys, color[exit_type] + '-', linewidth=0.5)
      axes[1].plot(xs, zs, color[exit_type] + '-', linewidth=0.5)

    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('z (mm)')
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')
    plt.show()

  inc_angles = np.linspace(args.scan_min, args.scan_max, args.scan_steps)
  frac_pass = np.zeros(len(inc_angles))
  frac_entr = np.zeros(len(inc_angles))

  r = np.max(np.abs(geom['top_xz'])) * 0.9
  for i, inc_angle in enumerate(tqdm(inc_angles)):
    n_pass, n_entr = 0, 0

    x0s, z0s = [], []
    while len(x0s) < args.n_rays:
      xs_try = np.random.uniform(-r, r, args.n_rays)
      zs_try = np.random.uniform(-r, r, args.n_rays)
      for x_try, z_try in zip(xs_try, zs_try):
        if _point_in_convex_polygon(x_try, z_try, geom['top_xz']):
          x0s.append(x_try)
          z0s.append(z_try)
          if len(x0s) >= args.n_rays:
            break

    for x0, z0 in zip(x0s, z0s):
      _, _, _, exit_type = propagate_hex(x0, args.height - 1, z0, inc_angle, geom)
      if exit_type == 'on sensor':
        n_pass += 1
      elif exit_type == 'bounced back':
        n_entr += 1

    frac_pass[i] = n_pass / args.n_rays
    frac_entr[i] = n_entr / args.n_rays

  if not args.quiet:
    plt.plot(inc_angles, frac_pass, 'b.-', label='on sensor')
    plt.plot(inc_angles, frac_entr, 'r.-', label='bounced back')
    plt.xlabel('Incident angle (deg)')
    plt.ylabel('Fraction')
    plt.legend()
    plt.show()

  with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['inc_angle_deg', 'fraction_on_sensor', 'fraction_bounced_back'])
    for angle, frac_on_sensor, frac_bounced in zip(inc_angles, frac_pass, frac_entr):
      writer.writerow([angle, frac_on_sensor, frac_bounced])
