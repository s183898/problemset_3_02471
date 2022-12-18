#%%
import numpy as np
from scipy import io
from sklearn.linear_model import LassoCV

# Load the data from the .mat file
data = io.loadmat('data/problem3_1.mat')
n = data['n']
x = data['x']

# Create the DCT matrix
l = 2**9
Phi = np.zeros((len(n), l))
for i, m in enumerate(n):
    Phi[i, m] = np.cos((np.pi / (2*l)) * (2*m - 1) * i)

#%%
# Use cross-validation to find the optimal value of alpha
alphas = np.logspace(-4, 1, 500)
lasso = LassoCV(alphas=alphas, cv=20, max_iter = 1000)
lasso.fit(Phi, x)

# Estimate K, aj, and mj
lasso_coefficients = lasso.coef_
K = np.count_nonzero(lasso_coefficients)
aj = lasso.coef_[lasso_coefficients != 0]
mj = np.where(lasso_coefficients != 0)[0]
opt_alpha = lasso.alpha_

# Print the results
print("K = ", K)
print("aj = ", aj)
print("mj = ", mj)
print("alpha = ", opt_alpha)

#%% evaluate the optimal model
from sklearn.metrics import mean_squared_error

# Reconstruct the original signal using the synthesis formula
x_reconstructed = np.dot(Phi, lasso_coefficients) 

# Calculate the mean squared error between the original and reconstructed signals
mse = mean_squared_error(x, x_reconstructed)
print("Mean squared error:", mse)

#compare the MSE to the variance of the original signal
print("Variance of the original signal:", np.var(x))

# if the MSE is much smaller than the variance, then this is an indication that the model is overfitting

#%% visualize the lasso coefficients
import matplotlib.pyplot as plt

plt.figure()
plt.stem(lasso_coefficients)
plt.xlabel('index j')
plt.ylabel('coefficient values a_j')
plt.title('Lasso coefficients')
plt.show()
#save figure as pdf
plt.savefig('plots/lasso_coefficients.pdf')

# %% visualize the original and reconstructed signals
plt.figure()
plt.plot(n,x,'o',label='original')
plt.plot(n, x_reconstructed, 'o' ,label='reconstructed')
plt.xlabel('index j')
plt.ylabel('signal values x_j')
plt.title('Original and reconstructed signals')
plt.legend()
plt.show()
#save figure as pdf
plt.savefig('plots/original_reconstructed.pdf')
