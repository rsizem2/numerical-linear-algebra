'''
Implements Algorithm 8.1: Modified Gram-Schmidt
Only works correctly for full rank matrices
'''
import numpy as np

def mgs(A):
    '''
    A - numpy array with linearly independent columns
    Q - orthogonal matrix whose columns span the range of mat
    R - upper diagnonal matrix such that A = QR
    '''
    V = A.copy()
    m,n = np.shape(V)
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i]/R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.dot(Q[:,i],V[:,j])
            V[:,j] = V[:,j] - Q[:,i]*R[i,j]
    return Q, R

# A = np.array([[1,2],[3,4],[5,9]])
# Q,R = mgs(A)
