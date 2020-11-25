# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:03:48 2020

@author: haixiang_zhang
"""

import math
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

# Generate the model
# Dimension and scale
params["d"] = 1
# params["N"] = int(sys.argv[1])
params["N"] = 150
# sub-Gaussian parameter
params["sigma"] = 3e0

# Optimality criteria
params["eps"] = 5e-1
params["delta"] = 1e-6

# Record average simulation runs and optimality gaps
total_samples = np.zeros((2,))
gaps = np.zeros((2,))
rate = np.zeros((2,))

# Open the output file
f_out = open("./results/sep_" + str(params["N"]) + ".txt", "w")

for t in range(1):
    print(t)
    model = models.separable_model.SeparableModel(params)
    
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
    
    # Use adaptive sampling algorithm
    output_ada = solvers.adaptive_solver.AdaptiveSolver(model["F"],params)
    print(output_ada)
    # Use uniform sampling algorithm
    output_uni = solvers.uniform_solver.UniformSolver(model["F"],params)
    print(output_uni)
    
    # Update records
    total_samples[0] = ( total_samples[0] * t + output_ada["total"] ) / (t+1)
    total_samples[1] = ( total_samples[1] * t + output_uni["total"] ) / (t+1)
    
    gaps[0] = ( gaps[0] * t + model["f"]([output_ada["x_opt"]]) - f_opt ) / (t+1)
    gaps[1] = ( gaps[1] * t + model["f"]([output_uni["x_opt"]]) - f_opt ) / (t+1)
    
    if model["f"]([output_ada["x_opt"]]) - f_opt <= params["eps"]:
        rate[0] = ( rate[0] * t + 1 ) / (t+1)
    else:
        rate[0] = ( rate[0] * t + 0 ) / (t+1)
    
    if model["f"]([output_uni["x_opt"]]) - f_opt <= params["eps"]:
        rate[1] = ( rate[1] * t + 1 ) / (t+1)
    else:
        rate[1] = ( rate[1] * t + 0 ) / (t+1)
    
    # # Use truncated subgradient descent method
    # output_grad = solvers.gradient_solver.GradientSolver(model["F"],params)
    # print(output_grad)
    
    f_out.write(" ".join( [str(output_ada["total"]),str(output_uni["total"]),\
                        str(model["f"]([output_ada["x_opt"]]) - f_opt),\
                        str(model["f"]([output_uni["x_opt"]]) - f_opt)] ))
    f_out.write("\n")
    f_out.flush()

f_out.write("\n")
f_out.write( " ".join([ str(total_samples[0]),str(gaps[0]),str(rate[0]) ]) )
f_out.write("\n")
f_out.write( " ".join([ str(total_samples[1]),str(gaps[1]),str(rate[1]) ]) )

f_out.close()



# stat = np.zeros((10,8))
# min_val = np.zeros((10,))

# for N in range(50,160,10):
#     i = 0
#     results = np.zeros((102,4))
        
#     with open("./results/sep_"+str(N)+".txt") as f_in:
#         for line in f_in:
#             line = line.split(" ")
#             if len(line) == 4:
#                 for j in range(4):
#                     results[i,j] = float(line[j])
#                 i += 1
#             elif len(line) == 3:
#                 for j in range(3):
#                     results[i,j] = float(line[j])
#                 i += 1
    
#     # Average simulation runs
#     stat[int(N/10)-1,0] = np.mean(results[:-2,0])
#     stat[int(N/10)-1,1] = np.std(results[:-2,0])
#     stat[int(N/10)-1,2] = np.mean(results[:-2,1])
#     stat[int(N/10)-1,3] = np.std(results[:-2,1])
    
#     # Average optimality gaps
#     stat[int(N/10)-1,4] = np.mean(results[:-2,2])
#     stat[int(N/10)-1,5] = np.std(results[:-2,2])
#     stat[int(N/10)-1,6] = np.mean(results[:-2,3])
#     stat[int(N/10)-1,7] = np.std(results[:-2,3])
    
#     # Minimal value
#     min_val[int(N/10)-1] = np.min( results[:-2,[2,3]] )

# plt.figure()
# plt.errorbar([N for N in range(10,110,10)], stat[:,0],
#              yerr=stat[:,1])
# plt.errorbar([N for N in range(10,110,10)], stat[:,2],
#              yerr=stat[:,3])
# # N_set = np.array([N for N in range(10,110,10)])
# # plt.plot( N_set, (np.log(N_set) **1) * 1e4 + stat[0,0] - 1e3*math.log(10)**2 )
# # plt.plot( N_set, np.log(N_set) * 6e2 + stat[1,2] - 6e2*math.log(20) )

# plt.figure()
# plt.errorbar([N for N in range(10,110,10)], stat[:,4],
#               yerr=stat[:,5])
# plt.errorbar([N for N in range(10,110,10)], stat[:,6],
#               yerr=stat[:,7])




