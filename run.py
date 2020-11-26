# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(1001)
import matplotlib.pyplot as plt

import models
import utils
import solvers

# Set parameters
params = {}

# Dimension and scale
params["d"] = 3
params["N"] = 1000

# Optimality criteria
params["eps"] = 5e-1
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1e0 # sub-Gaussian parameter
# model = models.queueing_model.QueueModel(params)
# model = models.quadratic_model.QuadraticModel(params)
model = models.separable_model.SeparableModel(params)

if "L" in model:
    params["L"] = model["L"]

# if "f" in model:
#     # Plot the function
#     x = np.linspace(1,params["N"],params["N"])
#     y = np.zeros((params["N"],))
    
#     for i,z in enumerate(x):
#         y[i] = model["f"]([z])
    
#     plt.plot(x,y)
# else:
#     # Plot the function
#     x = np.linspace(1,params["N"],params["N"])
#     y = np.zeros((params["N"],))
    
#     for i,z in enumerate(x):
#         for _ in range(20):
#             y[i] += model["F"]([z])
#     y /= 20
    
#     plt.plot(x,y)

# # Use adaptive sampling algorithm
# output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
# print(output_ada)
# # Use uniform sampling algorithm
# output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
# print(output_uni)

# Use truncated subgradient descent method
output_grad = solvers.gradient_solver.GradientSolver(model["f"],params)
print(output_grad)

# # Use Vaidya's cutting-plane method
# output_vai = solvers.vaidya_solver.VaidyaSolver(model["F"],params)
# print(output_vai)

# # Use cutting-plane method based on random walk
# output_random = solvers.random_walk_solver.RandomWalkSolver(model["F"],params)
# print(output_random)

# # Use dimension reduction method
# output_reduction = solvers.dim_reduction_solver.DimensionReductionSolver(model["F"],params)
# print(output_reduction)












