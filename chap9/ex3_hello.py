"""
Exercise 3 from Lecture 9
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Part (a) - Create "HELLO" matrix with 1's and 0's
'''

A = np.zeros((15,40))

def vertical (matrix, row, col):
    matrix[row:row+8][:,col:col+2] = 1
    return matrix

def horizontal (matrix, row, col):
    matrix[row:row+2][:,col:col+6] = 1
    return matrix

# Letter H

A = vertical(A,1,1)
A = horizontal(A,4,1)
A = vertical(A,1,5)

# Letter E

A = vertical(A,2,9)
A = horizontal(A,2,9)
A = horizontal(A,5,9)
A = horizontal(A,8,9)

# Letter L

A = vertical(A,3,17)
A = horizontal(A,9,17)

# Letter L

A = vertical(A,4,25)
A = horizontal(A,10,25)

# Letter O

A = vertical(A,5,32)
A = horizontal(A,5,32)
A = horizontal(A,11,32)
A = vertical(A,5,37)

plt.spy(A)
plt.show()

'''
Part (b) - SVD
'''

U,S,V = np.linalg.svd(A)

plt.scatter(x=range(15),y=S)
plt.title("Singular Values (linear scale)")
plt.show()

plt.semilogy(S, marker = "o", linestyle = "None")
plt.title("Singular Values (log scale)")
plt.show()

'''
Part (c) - Low Rank Approximations
'''

def low_rank_approx(M,rank):
    U,S,V = np.linalg.svd(M)
    X = np.zeros(M.shape)
    for idx, (u,v) in enumerate(zip(U.T,V)):
        if idx > rank: return X
        X += S[idx]*np.outer(u,v)
    return X

# Rank 1 Approximation
X1 = low_rank_approx(A,1)
plt.pcolor(np.flipud(X1), cmap = 'gray')
plt.title("Rank 1 Approximation")
plt.show()
    
# Rank 3 Approximation
X3 = low_rank_approx(A,3)
plt.pcolor(np.flipud(X3), cmap = 'gray')
plt.title("Rank 3 Approximation")
plt.show()

# Rank 5 Approximation
X5 = low_rank_approx(A,5)
plt.pcolor(np.flipud(X5), cmap = 'gray')
plt.title("Rank 5 Approximation")
plt.show()

# Rank 7 Approximation
X7 = low_rank_approx(A,7)
plt.pcolor(np.flipud(X7), cmap = 'gray')
plt.title("Rank 7 Approximation")
plt.show()

# Rank 9 Approximation
X9 = low_rank_approx(A,9)
plt.pcolor(np.flipud(X9), cmap = 'gray')
plt.title("Rank 9 Approximation")
plt.show()