# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:24:50 2020

@author: haixiang_zhang

Adaptive sampling algorithm for one-dim problems
"""

import math
import numpy as np
import time
from utils.subgaussian import required_samples, confidence_interval


def adaptive_solver(F, params):
    """
    The Adaptive sampling algorithm for one-dim problems.
    """

    # Retrieve parameters
    if "d" in params and params["d"] != 1:
        print("Adaptive sampling only works for one-dim problems.")
        return None
    N = params["N"] if "N" in params else 2
    # sigma = params["sigma"] if "sigma" in params else 1
    eps = params["eps"] if "eps" in params else 1e-1
    delta = params["delta"] if "delta" in params else 1e-6

    # Initial bounds
    L, U = 1, N
    # Total iterations
    T_max = math.log(N, 1.5) + 2

    # Start timing
    start_time = time.time()
    # Count simulation runs
    total_samples = 0

    while U - L > 2:
        # 3-quantiles
        N_1 = math.floor(2 * L / 3 + U / 3)
        N_2 = math.ceil(L / 3 + 2 * U / 3)

        # Upper bound on samples needed
        num_samples = required_samples(delta / 2 / T_max, eps / 8, params)
        # Empirical mean
        hat_F_1 = 0
        hat_F_2 = 0

        # Simulation
        for i in range(num_samples):
            hat_F_1 = (hat_F_1 * i + F([N_1])) / (i + 1)
            hat_F_2 = (hat_F_2 * i + F([N_2])) / (i + 1)

            # Check conditions
            CI = confidence_interval(delta / 2 / T_max, params, i + 1)

            # Condition (i)
            if hat_F_1 - hat_F_2 > 2 * CI:
                # Update total simulations
                total_samples += (2 * i)
                break
            # Condition (ii)
            elif hat_F_1 - hat_F_2 < -2 * CI:
                # Update total simulations
                total_samples += (2 * i)
                break

        # Condition (i)
        if hat_F_1 - hat_F_2 > 2 * CI:
            L = N_1
        # Condition (ii)
        elif hat_F_1 - hat_F_2 < -2 * CI:
            U = N_2
        # Condition (iii)
        else:
            # Update total simulations
            total_samples += (2 * num_samples)
            L, U = N_1, N_2

    # Solve the sub-problem
    hat_F = np.zeros((U - L + 1,))
    # Upper bound on samples needed
    num_samples = required_samples(delta / 2 / T_max, eps / 4, params)
    # Stop simulating if already too large
    blocked = np.zeros((U - L + 1,))

    # Simulation
    for i in range(num_samples):
        for j in range(U - L + 1):
            if blocked[j] == 0:
                hat_F[j] = (hat_F[j] * i + F([L + j])) / (i + 1)
        # Update total samples
        total_samples += np.sum(1 - blocked)

        # Check confidence interval
        CI = confidence_interval(delta / 2 / T_max, params, i + 1)
        # Block points with large empirical means
        blocked[hat_F - np.min(hat_F) > 2 * CI] = 1
        hat_F[hat_F - np.min(hat_F) > 2 * CI] = np.inf
        # Only one point left
        if np.sum(blocked) == U - L:
            break

    # Stop timing
    stop_time = time.time()
    # Return the point with the minimal empirical mean
    x_opt = np.argmin(hat_F) + L

    return {"x_opt": x_opt, "time": stop_time - start_time, "total": total_samples}
