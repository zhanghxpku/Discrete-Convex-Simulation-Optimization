# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np

np.random.seed(101)

import models
import solvers

# Set parameters
params = {}
# Dimension and scale
params["d"] = 4
params["N"] = 20

# Optimality criteria
params["eps"] = 5e4 / (params["d"] + 1)
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1e1  # sub-Gaussian parameter
model = models.queueing_model.queue_model(params)

if "N" in model:
    params["N"] = model["N"]

# Lipschitz constant and closed-form objective function
if "L" in model:
    params["L"] = model["L"]
    params["closed_form"] = True
else:
    params["L"] = 1
    params["closed_form"] = False

# Use adaptive sampling algorithm
output_ada = solvers.adaptive_solver.adaptive_solver(model["F"], params)
print(output_ada)
# Use uniform sampling algorithm
output_uni = solvers.uniform_solver.uniform_solver(model["F"], params)
print(output_uni)
# Use lil'UCB algorithm
output_ucb = solvers.ucb_solver.lilucb_solver(model["F"], params)
print(output_ucb)
