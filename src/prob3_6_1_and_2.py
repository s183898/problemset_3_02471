import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# 3.6.1 and 3.6.2

# Load .mat file
mat_file = scipy.io.loadmat('src/data/problem3_6.mat')

# Extract data from .mat file and create t
x = mat_file["t"].T
y = mat_file["y"][0]

diff = x[1] - x[0]
sample_multiplier = 20

# For ease of use t is sampled as an inverse multiple of the original sampling rate
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
print(f"Kernel Ridge argmax: {kernel_ridge_argmax:.2f}")

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.plot(t, y_pred, label='reconstructed signal', color = 'green')
ax.scatter(x, y, label='original signal', color='black', s=10)
ax.set_xlabel("Time (t)")
ax.set_ylabel("y(t)")
ax.set_title("Kernel Ridge Regression fit, C = 0.2, sigma = 0.5")
ax.axvline(kernel_ridge_argmax, label = f'Kernel Ridge argmax = {kernel_ridge_argmax:.2f}', color = 'blue', linestyle = 'dashed')
ax.axvline(5.1, label = 'Eyeball argmax = 5.1', color = 'red', linestyle = 'dashed')

ax.legend()
x_ticks = np.append(ax.get_xticks(), kernel_ridge_argmax)
ax.set_xticks(x_ticks)
ax.set_aspect('auto')
ax.set_xlim([-0.2, 12.6])
ax.set_xbound(lower=-0.2, upper=12.8)

plt.show()

