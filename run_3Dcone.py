#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm

import sys
sys.path.append('python')
from ConeProfile import *

tol = 1e-7 ## Numerical tolerance

def findSegments(x0, y0, z0, vx, vy, vz, mx, my):
  dmx = mx[1:]-mx[:-1]
  dmy = my[1:]-my[:-1]

def propagate(x0, y0, z0, theta, mirrors, sensor=None, n_bounces=20):
  theta = np.deg2rad(theta-90) ## initial light ray of theta=0 points downwards
  vx, vy, vz = np.cos(theta), np.sin(theta), 0 ## set the initial direction of light ray. we don't consider z-component (through the screen)
  xs, ys, zs = [x0], [y0], [z0]
  exit_type = 'bounce limit'

  rmin, rmax, ymax = None, None, None

  mxs, mys = [], []
  for mirror in mirrors:
    mx = np.array(mirror['x'])
    my = np.array(mirror['y'])
    mxs.append(mx)
    mys.append(my)

    rmax = mx.max() if rmax == None else max(rmax, np.abs(mx.min()), mx.max())
    ymax = my.max() if ymax == None else max(ymax, my.max())

  if sensor:
    sx = np.array(sensor['x'])
    sy = np.array(sensor['y'])

    rmax = sx.max() if rmax == None else max(rmax, np.abs(sx.min()), sx.max())
    ymax = sy.max() if ymax == None else max(ymax, sy.max())

  while n_bounces >= 0:
    bestR, bestX, bestY, bestZ = None, None, None, None
    bestType = None
    bestTangent = None
    for mx, my in zip(mxs, mys):
      x, y, z, dx, dy, dz = findSegments(x0, y0, z0, vx, vy, vz, mx, my)
      if len(x) > 0:
        r = getDist(x0, y0, z0, vx, vy, vz, x, y, z)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'mirror'
          bestTangent = [dx[irmin], dy[irmin], dz[irmin]]
    if sensor:
      x, y, z, dx, dy, dz = findSegments(x0, y0, z0, vx, vy, vz, sx, sy)
      if len(x) > 0:
        r = getDist(x0, y0, z0, vx, vy, vz, x, y, z)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, bestX, bestY, bestZ = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'on sensor'
    ## Add a virtual layer to pick up rays escaping backwards
    if bestType == None:
      x, y, z, dx, dy, dz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([-rmax, rmax]), np.array([ymax, ymax]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, vx, vy, vz, x, y, z)
        irmin = r.argmin()
        if bestR == None or r[irmin] < bestR:
          bestR, x0, y0, z0 = r[irmin], x[irmin], y[irmin], z[irmin]
          bestType = 'bounced back'
    else:
      x0, y0, z0 = bestX, bestY, bestZ
    if bestType == None:
      x, y, z, dx, dy, dz = findSegments(x0, y0, z0, vx, vy, vz,
                                         np.array([-rmax, rmax]), np.array([0, 0]))
      if len(x) > 0:
        r = getDist(x0, y0, z0, vx, vy, vz, x, y, z)
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

  par_n_rays = 101
  inc_angle = -20

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
    z0 = 0
    xs, ys, zs, exit_type = propagate(x0, par_height - 1, z0, inc_angle, config['mirrors'], sensor=config['sensor'])
    color = {'exit':'b', 'bounced back':'r', 'bounce limit':'r', 'on sensor':'g'}
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.plot(xs, ys, color[exit_type]+'-', linewidth=0.5)

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2 * rmax, 1.2 * rmax)
  plt.gca().set_aspect('equal', adjustable='box')
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
