import numpy as np

def ICAerr(Ahat, A):
    """
    Calculate the estimation error, i.e. the error between the estimated mixing matrix (Ahat) and the true mixing 
    matrix (A).
    The error between each vector and its estimate is computed as the 2-norm of their difference.
    The estimation error is computed as the 2-norm magnitude of the vector of errors.
    Finally, since we do not know the ordering of the vectors, we compute the error for both orderings and use
    the minimum as the estimation error.
    """
    e1 = np.linalg.norm(np.array([np.linalg.norm(Ahat[0] - A[0]),
                                  np.linalg.norm(Ahat[1] - A[1])]
                                )
                       )
    e2 = np.linalg.norm(np.array([np.linalg.norm(Ahat[0] - A[1]),
                                  np.linalg.norm(Ahat[1] - A[0])]
                                )
                       )
    return min(e1, e2)
