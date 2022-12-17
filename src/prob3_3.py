#%%
import numpy as np
import matplotlib.pyplot as plt
from ICA import ICA
from ICAerr import ICAerr

# mixing matrix
A = np.array([[3, 1], [1, 1]]) 

# normalize (true) mixing matrix
An = np.divide(A, np.max(A))

# number of observations
N = 5000

# set ICA parameters
mu = 0.1
components = 2
iterations = 200

# Experiments
num_experi = 100
e = np.zeros((num_experi ,2))

Distribs = ['uniform', 'uniform/beta', 'uniform/normal', 'multivariate normal']
distrib = Distribs[0]
print("Selected data distribution(s): " + distrib)

# calculate error of ICA estimate in 100 experiments, plot data for first experiment
for i in range(num_experi):
    if i > 0 and i % 10 == 0:
        print("Experiment " + str(i) + "/100")

    # generate data
    if distrib=='uniform':
        s = np.random.rand(N, 2) # s uniform distribution U(0,1)
    elif distrib == 'uniform/uniform':
        s = np.concatenate((np.random.rand(N, 1), 
                            np.random.beta(0.1, 0.1, size=(N,1))), 
                           axis=1) # s1 U(0,1), s2 beta distribution B(0.1, 0.1)
    elif distrib == 'uniform/normal':
        s = np.concatenate((np.random.rand(N, 1), 
                            np.random.normal(size=(N,1))), 
                           axis=1) # s1 U(0,1), s2 Gaussian distribution N(0, 1)
    else:
        s = np.random.multivariate_normal(mean=np.array([0.0, 1.0]), 
                                          cov=np.array([[2.0, 0.25], [0.25, 1.0]]), 
                                          size=N) # s multivariate Gaussian distribution with 
                                                  # mu=(0, 1), Sigma=[2 0.25; 0.25 1]

    # plot source data
    if i == 0:
        plt.figure(figsize=(10,10))
        plt.plot(s[:,0],s[:,1],'.')
        plt.show()

    # generate observations
    x = (A@s.T).T 

    # plot generated data
    if i == 0:
        plt.figure(figsize=(10,10))
        plt.plot(x[:,0],x[:,1],'.')
        plt.show()

    # center the data by taking the mean across the first (column) axis
    x = x - np.mean(x, axis=0)

    # run ICA
    break_ = False
    while(not break_):
        W = ICA(x, mu, components, iterations, 'subGauss')
        if np.linalg.det(W) >= 0:
            break_ = True # only accept ICA output if it has positive determinant

    # normalize unmixing matrix
    Wn = np.divide(W, np.max(W))

    # invert (normalized) unmixing matrix to obtain estimate of the mixing matrix 
    Ahat = np.linalg.inv(Wn)
    Ahatn = np.divide(Ahat, np.max(Ahat)) # normalize

    # display found estimate for comparison
    if i == 0:
        print("Normalized true A:\n", An)
        print("Normalized estimate:\n", Ahatn)

    # plot estimated sources (data projection on ica axis)
    if i == 0:
        z = (Wn@x.T).T # compute unmixed signals (estimated sources)
        plt.figure(figsize=(10,10))
        plt.plot(z[:,0],z[:,1],'.')
        plt.show()

    # compute error of ICA estimate
    e[i,:] = [i, ICAerr(Ahatn, An)]

# plot errors
plt.figure(figsize=(10,10))
#plt.plot(e[:,0],e[:,1],'.')
plt.hist(e[:,1])
plt.show()
