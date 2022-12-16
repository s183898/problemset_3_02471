import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import curve_fit

# Load .mat file
mat_file = scipy.io.loadmat('data/problem3_6.mat')

# Extract data from .mat file
t = mat_file["t"][0]
y = mat_file["y"][0]

def gkernel(x,y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

sigma = 0.6
mu = 5
C = 0.2

# Kernel Matrix

# K = np.zeros((len(t),len(t)))

# for i in range(len(t)):
#     for j in range(N):
#         K[i,j] = gkernel(x[j],t[i],sigma)

pair_dist = np.abs(t.reshape(-1, 1) - t.reshape(1, -1)) # solution  
K = np.exp(-1/(sigma**2)*pair_dist**2) # solution 
# print(K)

# Compute alpha
I = np.identity(K.shape[0])
theta = np.linalg.solve(K+C*I,y)

# Compute y_hat

# y_hat = (K+C*I)@theta

multi = 10
N = len(t)*multi
z0 = np.zeros(len(t))
x = np.linspace(0,12,N)

for k in range(t.shape[0]):
    z0[k] = 0
    for j in range(t.shape[0]):
        value = np.exp(-1/(sigma**2)*(x[k] - t[j])**2)
        z0[k] += theta[j]*value

zlist = np.zeros(N)

for i in range(N):
    kk = np.zeros(len(t)) 
    for j in range(len(t)):
        kk[j] = np.exp(-1/(sigma**2)*(x[i] - t[j])**2)

    zlist[i] = y.T@np.linalg.inv(K+C*I)@kk


# Power
zlist_undersampled = zlist[::multi]
power_s = np.mean(zlist**2)
power_n = np.mean((y-zlist_undersampled)**2)

# SNR

SNR = power_s/power_n
SNR_dB = 10*np.log10(SNR)

print("SNR: ", SNR)
print("SNR_dB: ", SNR_dB)

kernel_ridge_argmax = x[np.argmax(zlist)]

plt.plot(x, zlist, label='reconstructed signal')
plt.plot(t, y, 'b', label = 'signal')
# plt.plot(t,y_hat, 'r', label='y_hat', alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("y(t)")
plt.title("Problem 3.6")
plt.vlines(5.1, -0.7, 1.2, colors='r', linestyles='dashed', label='Eyeball argmax')
plt.vlines(kernel_ridge_argmax, -0.7, 1.2, colors='g', linestyles="dashed" , label='Kernel-Ridge argmax')
plt.legend()
plt.show()

