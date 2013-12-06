import numpy as np
from scipy import linalg as la

def QR(A):
    '''
    Compute the QR decomposition of an inveritble square matrix using the 
    modified Gram-Schmidt Algorithm. Assume no zeros occur on the main diagonal.
    Inputs:
        A -- square array of rank n, no zeros on main diagonal
    Return:
        (Q, R) -- A = QR, Q is orthonormal, R is upper triangular
    '''
    n = A.shape[0]
    U = np.copy(A) # make a copy of A so we don't change it
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    for i in xrange(n):
        r = ((U[:,i]**2).sum())**(.5)                # calculate r_{ii}
        q = (U[:,i]/r)                               # calculate q_i
        Q[:,i] = q                                  
        R[i,i] = r               
        rr = (U[:,i+1:]*(q.reshape(n,1))).sum(axis=0)  # caluclate r_{ij}, j>i
        U[:,i+1:] = U[:,i+1:] - rr*q.reshape(n,1)     # project out q_i
        R[i,i+1:] = rr
    return Q, R

def QRDet(A):
    '''
    Compute the magnitude of the determinant of a square invertible matrix using
    your QR decomposition function.
    Inputs:
        A -- square invertible matrix
    Return:
        the magnitude of the determinant of A.
    '''
    Q,R = QR(A)
    return np.diag(R).prod()

def LeastSquares(A,b):
    '''
    Using the scipy.linalg functions qr and solve_triangular, calculate
    the least sqaures solutions x_hat to the overdetermined system Ax = b.
    Inputs:
        A -- a full-rank (m,n) matrix
        b -- a m-dimensional vector
    Return:
        the least squares solution to the system Ax = b
    '''
    Q,R = la.qr(A, mode='economic')
    return la.solve_triangular(R,Q.T.dot(b))
