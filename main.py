# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(11)
import matplotlib.pyplot as plt

from gradient_descent import *

# Dimension and scale
d = 10
N = 10

# Generate a L-convex function
A = - np.abs(np.random.randn(d,d))
A = A + A.T

for i in range(d):
    A[i,i] = - np.sum(A[i,i+1:i+d]) + np.abs(np.random.randn(1,1))

b = np.random.randn(d)

# ell_inf Lipschitz constant
L = np.sqrt(d) * (np.linalg.norm(A,2) + np.linalg.norm(b))

# Objective function
f = lambda x: np.sum( x * (A @ x) ) + np.sum(b * x)
F = lambda x: f(x) + np.random.randn()

# The optimal solution
# x_opt = sgd_solver(d,N,f,L)

# Optimization ##########################################################
# Optimality criteria
eps = 1e0
delta = 1e-6

x_ssgd = ssgd_solver(d,N,f,F,L,eps,delta)
























