import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import curve_fit

# 3.6

# Load .mat file
mat_file = scipy.io.loadmat('data/problem3_6.mat')

# Extract data from .mat file and create t
x = mat_file["t"].T
y = mat_file["y"][0]

diff = x[1] - x[0]
sample_multiplier = 20

t = np.array([np.arange(min(x),max(x)+diff,diff/sample_multiplier)]).T
t_n = t.shape[0]
x_n = x.shape[0]


sigma = 0.5
C = 0.2

# Kernel Matrix
pair_dist = np.abs(x-x.T)
K = np.exp(-1/(sigma**2)*pair_dist**2)

# Compute alpha
I = np.identity(K.shape[0])
theta = np.linalg.solve(K+C*I,y)

y_pred = np.zeros(t_n)

for i in range(t_n):
    kernelVec = np.zeros(x_n) 
    for j in range(x_n):
        kernelVec[j] = np.exp(-1/(sigma**2)*(t[i] - x[j])**2)

    y_pred[i] = y.T@np.linalg.inv(K+C*I)@kernelVec

y_pred_sampled = y_pred[::sample_multiplier]

print("Check supersampling is done correctly:", np.allclose(y_pred_sampled, y_pred[::sample_multiplier], atol=1e-10))

# Power
power_s = np.mean(y_pred**2)
power_n = np.mean((y-y_pred_sampled)**2)

# SNR
SNR = power_s/power_n
SNR_dB = 10*np.log10(SNR)

print("SNR: ", SNR)
print("SNR_dB: ", SNR_dB)

kernel_ridge_argmax = t[np.argmax(y_pred)][0]

plt.plot(t, y_pred, label='reconstructed signal')
plt.scatter(x, y, label='data points', color='r', s=10)
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Problem 3.6")
plt.vlines(5.1, -0.7, 1.2, colors='r', linestyles='dashed', label='Eyeball argmax = 5.1')
plt.vlines(kernel_ridge_argmax, -0.7, 1.2, colors='g', linestyles="dashed" , label=f'Kernel-Ridge argmax = {kernel_ridge_argmax:.2f}')
plt.legend()
plt.show()
