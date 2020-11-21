# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt
import time

import models
import utils
import solvers

# Set parameters
params = {}

# Dimension and scale
params["d"] = 1
params["N"] = 1000

# Optimality criteria
params["eps"] = 1e0
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 10 # sub-Gaussian parameter
model = models.quadratic_model.QuadraticModel(params)

if "f" in model:
    # Plot the function
    x = np.linspace(1,params["N"],1000)
    y = np.zeros((1000,))
    
    for i,z in enumerate(x):
        y[i], _ = utils.lovasz.Lovasz(model["f"],[z],params)
    
    plt.plot(x,y)

# # Use adaptive sampling algorithm
time_start = time.time()
x_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
print(time.time() - time_start)
# Use uniform sampling algorithm
x_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
print(time.time() - time_start)

print(x_ada,x_uni)




















