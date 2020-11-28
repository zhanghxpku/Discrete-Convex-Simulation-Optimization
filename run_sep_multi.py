# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:13:51 2020

@author: haixiang_zhang
"""

import math
import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt

import models
import utils
import solvers

# print(sys.argv[1])

# Set parameters
params = {}

# Generate the model
# Dimension and scale
params["d"] = 9
# params["N"] = int(sys.argv[1])
params["N"] = 100
# sub-Gaussian parameter
params["sigma"] = 1e0

# Optimality criteria
params["eps"] = (math.factorial(params["d"]))**(1/params["d"]) / 5
params["delta"] = 1e-6

# Record average simulation runs and optimality gaps
total_samples = np.zeros((4,))
gaps = np.zeros((4,))
rate = np.zeros((4,))

# Open the output file
f_out = open("./results/sep_multi_" + str(params["d"]) + "_"\
             + str(params["N"]) + ".txt", "w")

for t in range(10):
    print(t)
    model = models.separable_model.SeparableModel(params)
    
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
    #         for _ in range(50):
    #             y[i] += model["F"]([z])
    #     y /= 50
    #     # plt.plot(x,y)
    
    # # Write the obj values
    # f_out.write(" ".join([ str(z) for z in y ]))
    # f_out.write("\n")
    
    # Get the optimal objective value
    f_opt = model["f"](model["x_opt"])
    
    # # Use truncated subgradient descent method
    output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
    # print(output_grad)
    # Use Vaidya's cutting-plane method
    # output_vai = solvers.vaidya_solver.VaidyaSolver(model["F"],params)
    # print(output_vai)
    # Use cutting-plane method based on random walk
    # output_random = solvers.random_walk_solver.RandomWalkSolver(model["F"],params)
    # print(output_random)
    # # Use dimension reduction method
    # output_reduction = solvers.dim_reduction_solver.DimensionReductionSolver(model["F"],params)
    # print(output_reduction)
    output_vai = output_grad
    output_random = output_vai
    output_reduction = output_grad
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_grad["total"] ) / (t+1)
    total_samples[1] = ( total_samples[1] * t + output_vai["total"] ) / (t+1)
    total_samples[2] = ( total_samples[2] * t + output_random["total"] ) / (t+1)
    total_samples[3] = ( total_samples[3] * t + output_reduction["total"] ) / (t+1)
    
    gaps[0] = ( gaps[0] * t + model["f"](output_grad["x_opt"]) - f_opt ) / (t+1)
    gaps[1] = ( gaps[1] * t + model["f"](output_vai["x_opt"]) - f_opt ) / (t+1)
    gaps[2] = ( gaps[2] * t + model["f"](output_random["x_opt"]) - f_opt ) / (t+1)
    gaps[3] = ( gaps[3] * t + model["f"](output_reduction["x_opt"]) - f_opt ) / (t+1)
    
    if model["f"](output_grad["x_opt"]) - f_opt <= params["eps"]:
        rate[0] = ( rate[0] * t + 1 ) / (t+1)
    else:
        rate[0] = ( rate[0] * t + 0 ) / (t+1)
    if model["f"](output_vai["x_opt"]) - f_opt <= params["eps"]:
        rate[1] = ( rate[1] * t + 1 ) / (t+1)
    else:
        rate[1] = ( rate[1] * t + 0 ) / (t+1)
    if model["f"](output_random["x_opt"]) - f_opt <= params["eps"]:
        rate[2] = ( rate[2] * t + 1 ) / (t+1)
    else:
        rate[2] = ( rate[2] * t + 0 ) / (t+1)
    if model["f"](output_reduction["x_opt"]) - f_opt <= params["eps"]:
        rate[3] = ( rate[3] * t + 1 ) / (t+1)
    else:
        rate[3] = ( rate[3] * t + 0 ) / (t+1)
    
    # # Use truncated subgradient descent method
    # output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
    # print(output_grad)
    
    f_out.write(" ".join( [str(output_grad["total"]),str(output_vai["total"]),\
                str(output_random["total"]),str(output_reduction["total"])] ))
    f_out.write("\n")
    f_out.write(" ".join( [str(model["f"](output_grad["x_opt"]) - f_opt),\
                        str(model["f"](output_vai["x_opt"]) - f_opt),\
                         str(model["f"](output_random["x_opt"]) - f_opt),\
                        str(model["f"](output_reduction["x_opt"]) - f_opt)] ))
    f_out.write("\n")
    f_out.flush()

f_out.write("\n")
f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]),str(rate[0]) ]) )
f_out.write("\n")
f_out.write( " ".join([ str(total_samples[1]),str(gaps[1]),str(rate[1]) ]) )
f_out.write("\n")
f_out.write( " ".join([ str(total_samples[2]),str(gaps[2]),str(rate[2]) ]) )
f_out.write("\n")
f_out.write( " ".join([ str(total_samples[3]),str(gaps[3]),str(rate[3]) ]) )

f_out.close()

