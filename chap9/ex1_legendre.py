"""
Experiment 1: Vandermonde Matrix/Legendre Polynomials
Adapted from MATLAB code in Lecture 9
"""

import numpy as np
import matplotlib.pyplot as plt


# Approximate Legendre Polynomials

x = np.linspace(start = -1, stop = 1, num = 257)                # discretization of (-1,1)
A = np.column_stack(((x**0).T,(x**1).T,(x**2).T,(x**3).T))      # construct Vandermonde Matrix
Q,R = np.linalg.qr(A)                                           # Reduced QR Factorization

scale = Q[-1,:]
Q = Q @ np.diag(1/scale)
plt.plot(x,Q)
plt.show()

# Error Calculation

L = np.column_stack(((x**0).T,
                     (x**1).T,
                     (1.5*(x**2).T-.5),
                     (2.5*((x**3).T)-1.5*((x**1).T))))
E = Q-L
plt.plot(x,E)
plt.show()

# Max Error vs Grid Size

def max_error(power):
    x = np.linspace(start = -1, stop = 1, num = 2**power+1)         
    A = np.column_stack(((x**0).T,(x**1).T,(x**2).T,(x**3).T))    
    Q,R = np.linalg.qr(A) 

    scale = Q[-1,:]
    Q = Q @ np.diag(1/scale)

    L = np.column_stack(((x**0).T,
                         (x**1).T,
                         (1.5*(x**2).T-.5),
                         (2.5*((x**3).T)-1.5*((x**1).T))))
    return np.abs(Q-L).max()

x = range(8,16)
y = map(max_error, x)

plt.plot(list(x),list(y))
plt.xlabel("Grid Size (2^x ticks)")
plt.ylabel("Absolute Error")
plt.show()

