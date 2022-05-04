# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:51 2020

@author: haixiang_zhang

Uniform sampling algorithm for one-dim problems
"""

import numpy as np
import time


def lilucb_solver(F, params):
    """
    The lil'UCB algorithm for one-dim problems.
    """

    # Retrieve parameters
    if "d" in params and params["d"] != 1:
        print("lil'UCB only works for one-dim problems.")
        return None
    N = params["N"] if "N" in params else 2
    sigma = params["sigma"] if "sigma" in params else 1
    eps0 = params["eps"] if "eps" in params else 1e-1
    nu = params["delta"] if "delta" in params else 1e-6

    # Parameters of the algorithm
    beta = 1
    a = 9
    eps = 1e-2
    delta = (nu * eps / 5 / (2 + eps)) ** (1 / (1 + eps))

    # Number of samples on each arm
    cur_samples = np.zeros((N,))
    # Empirical mean
    hat_F = np.zeros((N,))

    # Start timing
    start_time = time.time()

    # Initialize with one sample on each arm
    for i in range(N):
        hat_F[i] = -F([i + 1])
        cur_samples[i] = 1

    while (1 + a) * np.max(cur_samples) < 1 + a * np.sum(cur_samples):

        # Find the arm with maximal UCB
        deviation = np.log(np.log((1 + eps) * cur_samples + 2) / delta) / cur_samples
        deviation = (1 + beta) * (1 + np.sqrt(eps)) * np.sqrt(2 * (1 + eps) * deviation * sigma)
        I_t = np.argmax(hat_F + deviation)

        # Sample I_t
        hat_F[I_t] = (hat_F[I_t] * cur_samples[I_t] - F([I_t + 1])) \
            / (cur_samples[I_t] + 1)
        cur_samples[I_t] += 1

        # Check the LS criterion
        bound = np.log(2 * N * np.log((1 + eps) * cur_samples + 2) / delta) / cur_samples
        bound = (1 + np.sqrt(eps)) * np.sqrt(2 * (1 + eps) * bound * sigma)

        i_max = np.argmax(hat_F)
        lower = hat_F[i_max] - bound[i_max]
        if i_max < N - 1 and i_max > 0:
            upper = np.max([np.max(hat_F[:i_max] + bound[:i_max]),
                            np.max(hat_F[i_max + 1:] + bound[i_max + 1:])])
        elif i_max == N - 1:
            upper = np.max(hat_F[:i_max] + bound[:i_max])
        else:
            upper = np.max(hat_F[i_max + 1:] + bound[i_max + 1:])

        if lower >= upper - eps0 / 2:
            break

    # Return the point with the minimal empirical mean
    x_opt = np.argmax(hat_F) + 1
    # Count simulation runs
    total_samples = np.sum(cur_samples)

    # Stop timing
    stop_time = time.time()

    return {"x_opt": x_opt, "time": stop_time - start_time, "total": total_samples}
