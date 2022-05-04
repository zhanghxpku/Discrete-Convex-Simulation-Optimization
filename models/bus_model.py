# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:35:29 2020

@author: haixiang_zhang
Generate bus scheduling examples.
"""

import numpy as np
from scipy import stats


def bus_model(params):
    """
    Generate the objective function
    """

    # Parameters: lambda
    lam = 10
    # Retrieve parameters: upper bound of time
    tau = params["N"] - 1 if "N" in params else 100
    # Retrieve parameters: number of buses
    d = params["d"] if "d" in params else 1
    # The optimal solution
    x_opt = np.linspace(tau / (d + 1), tau * d / (d + 1), d) + 1

    return {"F": lambda y: waiting_time(y, params, False),
            "f": lambda y: waiting_time(y, params, True),
            "L": lam * tau, "x_opt": x_opt}


def waiting_time(x, params, expectation=False):
    """
    Compute the waiting time
    """

    # Parameters: lambda
    lam = 10
    # Retrieve parameters: upper bound of time
    tau = params["N"] + 1 if "N" in params else 100
    # Retrieve parameters: number of buses
    d = params["d"] if "d" in params else 1

    # Length of intervals
    length = np.zeros((d + 1,))
    length[0] = x[0] - 1
    length[1:-1] = np.abs(np.diff(x))
    length[-1] = tau + 1 - x[-1]

    if expectation:
        return lam / 2 * np.sum(length ** 2)
    else:
        # Number of arrivals in each interval
        # try:
        n = stats.poisson.rvs(length, size=(d + 1,))
        # except ValueError:
        #     print(length)
        #     return None

        # Waiting time
        wait_time = 0

        # Compute the waiting time
        for i in range(d + 1):
            # # Number of arrivals
            # n = stats.poisson.rvs(length[i] * lam)
            # Waiting time
            wait_time += stats.uniform.rvs(0, n[i] * length[i] / 2)

        return wait_time
