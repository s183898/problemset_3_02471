import numpy as np

def ICAerr(Ahat, A):
    """
    Calculate the error between the estimated mixing matrix, Ahat, and the true mixing matrix, A.
    The contribution to the error of an estimated vector Ahat[i]
    The error contribution of between an estimated vector and a true vector is computed as their (2-norm) distance.
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
