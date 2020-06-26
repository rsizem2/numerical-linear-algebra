# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:16:48 2020

@author: rsizem2
"""

import numpy as np

def house(A):
    R = A.copy()
    m,n = A.shape
    W = np.zeros((m,n))
    for k in range(n):
        x = R[k:,k]
        e = np.zeros(x.shape)
        e[0,0] = 1
        v = np.sign(x[0])*np.linalg.norm(x)*e + x
        v = v/np.linalg.norm(v)
        W[k:,k] = v
        R[k:,k] = R[k:,k] - 2*v*(v.conj().T @ R[k:,k:])
    return W, R

def formQ(W):
    m,n = W.shape
    Q = np.eye(m)
    for col in range(m):
        for k in range(n-1,-1):
            v = W[k:,k]
            Q[:,col] = Q[:,col]-2*v*np.vdot(v,Q[:,col])
    return Q
    