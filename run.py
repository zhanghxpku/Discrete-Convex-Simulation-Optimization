# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 19:26:17 2020

@author: haixiang_zhang
"""

import math
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
params["eps"] = math.log(params["d"]+1,2) / 5
# params["eps"] = math.log(params["d"]+1,2)*params["N"]**0.5 / 80
params["eps"] = 1e0
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 1e1 # sub-Gaussian parameter
model = models.queueing_model.QueueModel(params)
# model = models.quadratic_model.QuadraticModel(params)
# model = models.separable_model.SeparableModel(params)

# Lipschitz constant and closed-form objective function
if "L" in model:
    params["L"] = model["L"]
    params["closed_form"] = True
else:
    params["L"] = 1
    params["closed_form"] = False

# if "f" in model:
#     # Plot the function
#     x = np.linspace(1,params["N"],params["N"])
#     y = np.zeros((params["N"],))
    
#     for i,z in enumerate(x):
#         y[i] = model["f"]([z])
    
#     s = np.ones(x.shape)
#     plt.scatter(x,y,s=s)
#     plt.xlabel("N")
#     plt.ylabel("Objestive value")
#     # plt.savefig("./results/sqrt_sep_new/obj.png",bbox_inches='tight', dpi=300)
# else:
#     # Plot the function
#     x = np.linspace(1,params["N"],params["N"])
#     y = np.zeros((params["N"],))
    
#     for i,z in enumerate(x):
#         for _ in range(100):
#             y[i] += model["F"]([z])
#     y /= 100
    
#     s = np.ones(x.shape)
#     plt.scatter(x,y,s=s)
#     plt.xlabel("N")
#     plt.ylabel("Average waiting time")
#     # plt.savefig("./results/queue/obj.png",bbox_inches='tight', dpi=300)

# # Use adaptive sampling algorithm
# output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
# print(output_ada)
# Use uniform sampling algorithm
output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
print(output_uni)
# Use lil'UCB algorithm
output_ucb = solvers.ucb_solver.LILUCBSolver(model["F"],params)
print(output_ucb)

# # Use truncated subgradient descent method
# output_grad = solvers.gradient_solver.GradientSolver(model["f"],params)
# print(output_grad)
# # Use Vaidya's cutting-plane method
# output_vai = solvers.vaidya_solver.VaidyaSolver(model["F"],params)
# print(output_vai)
# # Use cutting-plane method based on random walk
# output_random = solvers.random_walk_solver.RandomWalkSolver(model["F"],params)
# print(output_random)
# # Use dimension reduction method
# output_reduction = solvers.dim_reduction_solver.DimensionReductionSolver(model["F"],params)
# print(output_reduction)












