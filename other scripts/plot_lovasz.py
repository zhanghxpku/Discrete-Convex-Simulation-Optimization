# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:39:31 2021

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

# The scale of problem
N = 3
# # The number of points in each dimension in each square
# n = 1

# The objective function
A = -np.abs(np.random.normal(0,1,size=(2,2)))
# A = A + A.T
for i in range(2):
    A[i,i] = - 1.8 * A[i,1-i]
f = lambda x : x.T @ A @ x

X = np.arange(1,N+1,1)
Y = np.arange(1,N+1,1)
X,Y = np.meshgrid(X, Y)
Z = np.stack((X,Y),axis=-1)
W = np.sum( (Z@A) * Z, axis=-1 )
max_val = np.max(np.abs(W))

# Normalize the objective function
A /= max_val
f = lambda x : x.T @ A @ x
X = np.arange(1,N+0.1,0.1)
Y = np.arange(1,N+0.1,0.1)
X,Y = np.meshgrid(X, Y)
Z = np.stack((X,Y),axis=-1)
W = np.sum( (Z@A) * Z, axis=-1 )

ax = a3.Axes3D(pl.figure())
# ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,W,cmap=cm.coolwarm)

ax.set_xlim(1,N)
ax.set_ylim(1,N)
ax.set_zlim(0,1)
pl.xticks(np.arange(1, N+1, 1.0))
pl.yticks(np.arange(1, N+1, 1.0))
plt.savefig("lovasz_2.png",bbox_inches='tight', dpi=300)

# Plot the Lovasz extension
ax = a3.Axes3D(pl.figure())

for x in range(1,N):
    for y in range(1,N):
        base_node = np.array([x,y])
        # Compute the values of objective function
        val1 = f(np.array([x,y]))
        val2 = f(np.array([x+1,y]))
        val3 = f(np.array([x,y+1]))
        val4 = f(np.array([x+1,y+1]))

        vtx1 = np.array([ [x,y,val1],[x+1,y,val2],[x+1,y+1,val4] ])
        tri = a3.art3d.Poly3DCollection([vtx1])
        tri.set_color(colors.rgb2hex(cm.coolwarm((val1+val2+val4)/3)))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
        
        vtx2 = np.array([ [x,y,val1],[x,y+1,val3],[x+1,y+1,val4] ])
        tri = a3.art3d.Poly3DCollection([vtx2])
        tri.set_color(colors.rgb2hex(cm.coolwarm((val1+val3+val4)/3)))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

ax.set_xlim(1,N)
ax.set_ylim(1,N)
pl.xticks(np.arange(1, N+1, 1.0))
pl.yticks(np.arange(1, N+1, 1.0))
ax.set_zlim(0,1)
pl.savefig("lovasz_1.png",bbox_inches='tight', dpi=300)