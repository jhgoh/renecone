#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm

import sys
sys.path.append('python')
from ConeProfile import *

def propagate(x0, y0, z0, theta, mirrors, sensors=None, n_bounces=20):
  tol = 1e-9 ## Numerical tolerance

  ## endpoints of light ray segments
  xs = [x0]
  ys = [y0]
  zs = [z0]

  theta = np.deg2rad(theta-90) ## initial light ray of theta=0 points downwards
  vx, vy, vz = np.cos(theta), np.sin(theta), 0 ## set the initial direction of light ray. we don't consider z-component (through the screen)
  
  bounces = 0
  while True:
    ## We model the mirror and sensor surfaces by revolving piecewise linear segments in x-y plane along the y-axis,
    ## in other words, patch of straight cones sliced by some height, alighed along the y-axis.
    ## This semi-analytic approach will give a precise and fast result for the object with a cylindrical symmetry.
    ##
    ## The coordinate system could be somehow inconvenient (to me) - cone is pointing to the y-axis, not the z-axis nor the x-axis.
    ## but let us bear with it to keep the consistency of 2D cone

    ## For the first step, rotate along the y-axis where the line of the ray looks like a vertical line,
    ## so that we can break down the problem to series of 2D geometry problems
    msegs = []
    ## Exception: vy=0 -> we already know the y position of the next bounce
    if np.abs(vx) <= tol:
      ## Note: there are lots of efficient search algorithms, but we loop over all vertices for generallity
      for mxs, mys in mirrors:
        
        
      
    phi = np.atan2(vz, vx)
    for mxs, mys in mirrors:
      

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
  config['sensors'] = make_sensor(par_dout, sensor_curv=par_sensor_curv)
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

  ### Trace a few sample rays
  #for x0 in np.linspace(-config['din'] / 2 * 0.9, config['din'] / 2 * 0.9, par_n_rays):
  #  xs, ys, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensors=config['sensors'])
  #  color = {'exit':'b', 'bounced back':'r', 'bounce limit':'r', 'on sensor':'g'}
  #  plt.xlabel('x (mm)')
  #  plt.ylabel('y (mm)')
  #  plt.plot(xs, ys, color[exit_type]+'-', linewidth=0.5)

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
  #    _, _, exit_type = propagate(x0, par_height - 1, inc_angle, config['mirrors'], sensors=config['sensors'])
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
