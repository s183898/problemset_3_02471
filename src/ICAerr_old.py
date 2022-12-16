import numpy as np

# calculate error between estimated mixing matrix Ahat and true mixing matrix A
def ICAerr(Ahat, A):#, num_components):
    # for each vector in Ahat, find its closest match in A and regard the dist (2-norm) as error contribution
    e = np.zeros(num_components)
    for i in range(len(Ahat)):
        # find vector index in A that has minimum dist to estimated vector Ahat[i]
        j = np.argmin(np.linalg.norm(A - Ahat[i], axis=1))
        # compute error
        e[i] = np.linalg.norm(Ahat[i] - A[j]) 
    return e
