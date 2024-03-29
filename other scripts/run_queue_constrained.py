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
params["N"] = int(sys.argv[1])
params["M"] = int(sys.argv[2])
# params["N"] = 10
# params["M"] = 12
params["d"] = params["M"]
# params["N"] = params["scale"] * params["d"]
# Regularization constraint
params["c"] = 0 / params["M"]
params["K"] = int(params["N"] * params["d"] / 3)
params["trunc"] = bool(int(sys.argv[3]))
# params["eta"] = float(sys.argv[4])
params["eta"] = 1 if params["trunc"] else 0.1

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
f_out = open("./results/queue_cons_" + str(params["N"]) + "_" + str(params["M"]) + "_" + str(params["trunc"]) + ".txt", "w")

for t in range(5):
    print(t)
    model = models.queueing_or_model.queuereg_model_24hr(params)
    
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
    
    f_out.write(str(t))
    f_out.write("\n")
    f_out.flush()
    
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
    #         for _ in range(700):
    #             y[i] += model["F"]([z,z,z,z,z])
    #     y /= 700
    #     plt.plot(x,y)
    
#     # # Write the obj values
#     # f_out.write(" ".join([ str(z) for z in y ]))
#     # f_out.write("\n")
    
#     # # Get the optimal objective value
#     # f_opt = model["f"](model["x_opt"])
    
    # Use truncated subgradient descent method
    output_grad = solvers.gradient_solver.GradientProjSolver(model["F"],params,params["trunc"])
    print(output_grad)
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_grad["total"] ) / (t+1)
    # gaps[0] = ( gaps[0] * t + model["f"](output_grad["x_opt"]) - f_opt ) / (t+1)

    f_out.write(str(output_grad["total"]))
    f_out.write("\n")
    # f_out.write(str(model["f"](output_grad["x_opt"]) - f_opt))
    # f_out.write("\n")
    f_out.flush()
    
    sample = np.zeros((5000,))
    for i in range(5000):
        sample[i] = model["F"](output_grad["x_opt"])
    print(np.std(sample),np.mean(sample))

# # for t in range(1):
# #     if min_val - f_opt <= params["eps"]:
# #             rate[0] = ( rate[0] * t + 1 ) / (t+1)
# #         else:
# #             rate[0] = ( rate[0] * t + 0 ) / (t+1)

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
