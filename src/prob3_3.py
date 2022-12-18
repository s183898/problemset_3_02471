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

s_plot = np.zeros((4,N,2))
x_plot = np.zeros((4,N,2))
z_plot = np.zeros((4,N,2))

# Experiments
num_experi = 100
e = np.zeros((4,num_experi))

distribs = ['uniform', 'uniform/beta', 'uniform/normal', 'multivariate normal']
colors = ['red', 'green', 'orange', 'cyan']
titles = ['(a)', '(b)', '(c)', '(d)']

# calculate error of ICA estimate in 100 experiments, plot data for first experiment
for k in range(4):
    print("Data distribution(s): " + distribs[k])
    for i in range(num_experi):
        if i > 0 and i % 10 == 0:
            print("Experiment " + str(i) + "/100")

        # generate data
        if k == 0:
            s = np.random.rand(N, 2) # s uniform distribution U(0,1)
        elif k == 1:
            s = np.concatenate((np.random.rand(N, 1), 
                                np.random.beta(0.1, 0.1, size=(N,1))), 
                               axis=1) # s1 U(0,1), s2 beta distribution B(0.1, 0.1)
        elif k == 2:
            s = np.concatenate((np.random.rand(N, 1), 
                                np.random.normal(size=(N,1))), 
                               axis=1) # s1 U(0,1), s2 Gaussian distribution N(0, 1)
        else:
            s = np.random.multivariate_normal(mean=np.array([0.0, 1.0]), 
                                              cov=np.array([[2.0, 0.25], [0.25, 1.0]]), 
                                              size=N) # s multivariate Gaussian distribution with 
                                                      # mu=(0, 1), Sigma=[2 0.25; 0.25 1]

        # source data for later plotting
        if i == 0:
            s_plot[k,:,:] = s

        # generate observations
        x = (A@s.T).T 

        # store generated data for later plotting
        if i == 0:
            x_plot[k,:,:] = x

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

        # store estimated sources for later plotting
        if i == 0:
            z = (Wn@x.T).T # compute unmixed signals (estimated sources)
            z_plot[k,:,:] = z

        # compute error of ICA estimate
        e[k,i] = ICAerr(Ahatn, An)

# plot source data
fig, axes = plt.subplots(2,2, figsize=(10,10))
for k in range(4):
    if k == 0:
        coords = [0,0]
    elif k == 1:
        coords = [0,1]
    elif k == 2:
        coords = [1,0]
    else:
        coords = [1,1]

    axes[coords[0]][coords[1]].plot(s_plot[k,:,0], s_plot[k,:,1], '.', color=colors[k])
    axes[coords[0]][coords[1]].set_title(titles[k])
    if coords[0] == 1:
        axes[coords[0]][coords[1]].set_xlabel('s_1')
    if coords[1] == 0:
        axes[coords[0]][coords[1]].set_ylabel('s_2')

plt.show()

# plot observations
fig, axes = plt.subplots(2,2, figsize=(10,10))
for k in range(4):
    if k == 0:
        coords = [0,0]
    elif k == 1:
        coords = [0,1]
    elif k == 2:
        coords = [1,0]
    else:
        coords = [1,1]
    

    axes[coords[0]][coords[1]].plot(x_plot[k,:,0], x_plot[k,:,1], '.', color=colors[k])
    axes[coords[0]][coords[1]].set_title(titles[k])
    if coords[0] == 1:
        axes[coords[0]][coords[1]].set_xlabel('x_1')
    if coords[1] == 0:
        axes[coords[0]][coords[1]].set_ylabel('x_2')

plt.show()

# plot estimated sources (data projection on ICA axis)
fig, axes = plt.subplots(2,2, figsize=(10,10))
for k in range(4):
    if k == 0:
        coords = [0,0]
    elif k == 1:
        coords = [0,1]
    elif k == 2:
        coords = [1,0]
    else:
        coords = [1,1]

    axes[coords[0]][coords[1]].plot(z_plot[k,:,0], z_plot[k,:,1], '.', color=colors[k])
    axes[coords[0]][coords[1]].set_title(titles[k])
    if coords[0] == 1:
        axes[coords[0]][coords[1]].set_xlabel('z_1')
    if coords[1] == 0:
        axes[coords[0]][coords[1]].set_ylabel('z_2')
 
plt.show()

# plot error histograms
fig, axes = plt.subplots(2,2, figsize=(10,10))
for k in range(4):
    if k == 0:
        coords = [0,0]
    elif k == 1:
        coords = [0,1]
    elif k == 2:
        coords = [1,0]
    else:
        coords = [1,1]

    axes[coords[0]][coords[1]].hist(e[k,:], color=colors[k], edgecolor='black')
    axes[coords[0]][coords[1]].set_title(titles[k])
    if coords[0] == 1:
        axes[coords[0]][coords[1]].set_xlabel('Estimation error')
    if coords[1] == 0:
        axes[coords[0]][coords[1]].set_ylabel('Experiments')

plt.show()
