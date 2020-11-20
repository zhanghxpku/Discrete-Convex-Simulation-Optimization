# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt

import models
import utils
import solvers

# Set parameters
params = {}

# Dimension and scale
params["d"] = 1
params["N"] = 10

# Optimality criteria
params["eps"] = 1e0
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1 # sub-Gaussian parameter
model = models.quadratic_model.QuadraticModel(params)


# Use adaptive sampling algorithm
x_opt = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)



x = np.linspace(1,params["N"],100)
y = np.zeros((100,))

for i,z in enumerate(x):
    y[i], _ = utils.lovasz.Lovasz(model["F"],[z],params)

plt.plot(x,y)



















