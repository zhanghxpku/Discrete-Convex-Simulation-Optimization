# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:05:37 2021

@author: haixiang_zhang
"""

import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt
import sys

import models
import utils
import solvers

# print(sys.argv[1])

# Set parameters
params = {}

# Dimension and scale
# params["N"] = int(sys.argv[2])
# params["M"] = int(sys.argv[2])
params["N"] = 50
params["M"] = 4
params["d"] = params["M"]
# params["N"] = params["scale"] * params["d"]
# Regularization constraint
params["c"] = 10 / params["M"]
params["K"] = params["N"] * params["d"]
# params["K"] = int(params["N"] * params["d"] / 3)
# params["trunc"] = bool(int(sys.argv[3]))
params["trunc"] = True
# method = int(sys.argv[1])
method = 0

if method == 0:
    params["eta"] = 1 if params["trunc"] else 0.05
else:
    # params["eta"] = float(sys.argv[3])
    params["eta"] = 10

# Optimality criteria
params["eps"] = params["N"] / 2
params["delta"] = 1e-6

# Generate the model
params["sigma"] = 30 * np.sqrt(params["N"]) # sub-Gaussian parameter

# Record average simulation runs and optimality gaps
total_samples = np.zeros((2,))
gaps = np.zeros((2,))
rate = np.zeros((2,))

# Open the output file
f_out = open("./results/queue_uncons_" + str(params["N"]) + "_"  + str(params["d"]) + "_" + str(method) + "_" + str(params["eta"]) + ".txt", "w")

for t in range(1):
    print(t)
    model = models.queueing_or_model.QueueRegORModel(params)
    
    f_out.write(str(t))
    f_out.write("\n")
    f_out.flush()
    
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
        
    #     plt.plot(x,y)
    # else:
        # # Plot the function
        # x = np.linspace(1,params["N"],params["N"])
        # y = np.zeros((params["N"],))
        
        # for i,z in enumerate(x):
        #     for _ in range(700):
        #         y[i] += model["F"]([z])
        # y /= 700
        # plt.plot(x,y)
    
#     # # Write the obj values
#     # f_out.write(" ".join([ str(z) for z in y ]))
#     # f_out.write("\n")
    
#     # # Get the optimal objective value
#     # f_opt = model["f"](model["x_opt"])
    
    # Use truncated subgradient descent method
    if method == 0:
        output_grad = solvers.gradient_solver.GradientProjSolver(model["F"],params,params["trunc"])
    elif method == 1:
        output_grad = solvers.vaidya_solver.VaidyaProjSolver(model["F"],params)
    elif method == 2:
        output_grad = solvers.random_walk_solver.RandomWalkProjSolver(model["F"],params)
    elif method == 3:
        output_grad = solvers.dim_reduction_solver.DimensionReductionProjSolver(model["F"],params)
    elif method == 4:
        output_grad = solvers.gps_solver.GPSSolver(model["F"],params)
    print(output_grad)
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_grad["total"] ) / (t+1)

    f_out.write(str(output_grad["total"]))
    f_out.write("\n")
    # f_out.write(str(model["f"](output_grad["x_opt"]) - f_opt))
    # f_out.write("\n")
    f_out.flush()
    
    sample = np.zeros((5000,))
    for i in range(5000):
        sample[i] = model["F"](output_grad["x_opt"])
    print(np.std(sample),np.mean(sample))
    gaps[0] = ( gaps[0] * t + np.mean(sample) ) / (t+1)

f_out.write("\n")
f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]),str(rate[0]) ]) )

f_out.close()


# sample = np.zeros((5000,))
# for i in range(5000):
#     sample[i] = model["F"](output_grad["x_opt"])
# print(np.std(sample),np.mean(sample))

# sample = np.zeros((5000,))
# for i in range(5000):
#     sample[i] = model["F"](np.cumsum(params["N"] * np.ones((params["d"],))))
# print(np.std(sample),np.mean(sample))
