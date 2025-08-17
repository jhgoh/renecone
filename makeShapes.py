#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def makePlanar(din, dout, angle, width=1200, height=1200):
  result = {
    'name': 'planar',
    'nwalls': 4,
    'din': din,
    'dout': dout,
    'angle': angle,
    'shapes': [],
  }

  ## Left and right walls
  result['shapes'].append({'x':[-width/2, -width/2], 'y':[0, height]})
  result['shapes'].append({'x':[width/2, width/2], 'y':[0, height]})

  ## Main mirrors
  h = dout*np.tan(np.deg2rad(angle))
  result['shapes'].append({'x':[dout/2, din/2], 'y':[0, h]})
  result['shapes'].append({'x':[-dout/2, -din/2], 'y':[0, h]})

  return result

def makeWinston(din, dout, critAngle, width=1200, height=1200, nPoint=25):
  result = {
    'name': 'planar',
    'nwalls': 4,
    'din': din,
    'dout': dout,
    'critAngle': critAngle,
    'shapes': [],
  }

  ## Left and right walls
  result['shapes'].append({'x':[-width/2, -width/2], 'y':[0, height]})
  result['shapes'].append({'x':[width/2, width/2], 'y':[0, height]})

  ## Main mirrors
  critAngle = np.deg2rad(critAngle)
  rin = din/2
  rout = dout/2
  sinC, cosC = np.sin(critAngle), np.cos(critAngle)
  l = (rin + rout) / np.tan(critAngle) ## Intrinsic height
  f = rout*(1+sinC) ## Focal length

  ## Build parabola in mirror axis
  d1 = 2*rout*cosC
  d2 = l*np.sin(2*critAngle)/cosC
  xr = np.linspace(d1, d2, nPoint)
  yr = xr**2/4/f

  ## Rotate & translate the mirror
  xx = xr*cosC - (yr-f)*sinC - rout
  yy = xr*sinC + (yr-f)*cosC

  #yy = yy[xx<=rout]
  #xx = xx[xx<=rout]

  result['shapes'].append({'x':xx, 'y':yy})
  result['shapes'].append({'x':-xx, 'y':yy})

  return result

def propagate(x, y, vx, vy, mirrors, nstep=10000):
  for i in range(nstep):
    

if __name__ == '__main__':
  par_din = 1200
  par_dout = 500
  par_angle = 25
  #par_angle = 10

  #result = makePlanar(par_din, par_dout, par_angle)
  result = makeWinston(par_din, par_dout, par_angle)
  #print(result)

  rmax = 0
  for shape in result['shapes']:
    x, y = shape['x'], shape['y']
    x, y = np.array(x), np.array(y)
    plt.plot(x, y, 'k')

    rmax = max(np.hypot(x, y).max(), rmax)

  ## Draw axis
  plt.plot([-result['din']/2, result['din']/2], [0, 0], '-.k', linewidth=0.5)
  plt.plot([0, 0], [0, 1.5*rmax], '-.k', linewidth=0.5)

  plt.xlim(-rmax, rmax)
  plt.ylim(-0.2*rmax, 1.2*rmax)
  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()

