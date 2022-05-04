# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:02:33 2020

@author: haixiang_zhang

The LLL algorithm.
"""

import numpy as np


def LLL(basis, A):
    """
    Compute an LLL-reduced basis under A-norm.
    """

    # Copy the parameter
    b = np.copy(basis).astype('float64')
    # Get the dimension
    d = basis.shape[0]

    # Algorithm parameter
    delta = 0.75
    # Gram-Schmidt orthogonalization
    u = GramSchmidt(b, A)

    k = 1
    while k < d:
        for j in range(k - 1, -1, -1):
            mu_kj = inner_product(b[k, :], u[j, :], A) \
                    / inner_product(u[j, :], u[j, :], A)

            if abs(mu_kj) > 0.5:
                b[k, :] -= (np.round(mu_kj) * b[j, :])
                # Gram-Schmidt orthogonalization
                u = GramSchmidt(b, A)

        mu_kk_1 = inner_product(b[k, :], u[k - 1, :], A) \
            / inner_product(u[k - 1, :], u[k - 1, :], A)
        if inner_product(u[k, :], u[k, :], A) >= \
                (delta - mu_kk_1 ** 2) * inner_product(u[k - 1, :], u[k - 1, :], A):
            k += 1
        else:
            # Exchange basis
            b[[k - 1, k], :] = b[[k, k - 1], :]
            # Gram-Schmidt orthogonalization
            u = GramSchmidt(b, A)
            k = max(k - 1, 1)

    return b


def GramSchmidt(basis, A):
    """
    Gram-Schmidt orthogonalization under A-norm.
    """

    # Get the dimension
    d = basis.shape[0]

    # The orthogonal basis
    u = np.zeros(basis.shape)

    for i in range(d):
        u[i, :] = basis[i, :]
        for j in range(i):
            u[i, :] -= ((inner_product(u[i, :], u[j, :], A)
                         / inner_product(u[j, :], u[j, :], A)) * u[j, :])

    return u


def inner_product(x, y, A):
    """
    Inner product under A-norm.
    """

    return np.sum(x * (A @ y))
