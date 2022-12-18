import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import curve_fit
import librosa.display
from random import randrange
from sklearn.svm import SVR

# learning parameters
epsilon = 0.02
kernel_type = 'Gaussian'
kernel_params = 0.6
C = 1

mat_file = scipy.io.loadmat('data/problem3_6.mat')

# sample_multi = 8
diff = 0.2
sample_multi = 20
# Extract data from .mat file
x = np.array(mat_file["t"])
y = mat_file["y"][0]
t = np.array([np.arange(min(x[0]),max(x[0])+diff,diff/sample_multi)]).T

x = x.T

gamma = 1/(np.square(kernel_params)) 
regressor = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)

regressor.fit(x,y)
y_pred = regressor.predict(t)

y_pred_sampled = y_pred[::sample_multi]

print("Check supersampling is done correctly:", np.allclose(y_pred_sampled, y_pred[::sample_multi], atol=1e-10))

def outlier_detection(y,y_pred, regressor, epsilon, threshold):
    if threshold < 0:
        print("Warning. Threshold should be non-negative. Setting threshold to 0.")
        threshold = 0

    y_pred = regressor.predict(x)
    SE = np.abs(y - y_pred)
    out = SE > epsilon + threshold
    return out

outliers = outlier_detection(y,y_pred_sampled, regressor, epsilon = epsilon, threshold = 0.1)
inliers = np.logical_not(outliers)

# Power
power_s = np.mean(y[inliers]**2)
power_n = np.mean((y[inliers]-y_pred_sampled[inliers])**2)

# SNR
SNR = power_s/power_n
SNR_dB = 10*np.log10(SNR)

print("SNR: ", SNR)
print("SNR_dB: ", SNR_dB)

SVM_argmax = t[np.argmax(y_pred)]

# plot

fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.axvline(SVM_argmax, label = f'SVR argmax = {SVM_argmax[0]:.2f}', color = 'blue', linestyle = 'dashed')
ax.plot(t, y_pred, color = 'green', label = 'SVR fit')
ax.stem(x[regressor.support_], y[regressor.support_], linefmt = 'none', markerfmt='yo', label='support vector', basefmt=" ")
ax.stem(x, y,  linefmt = 'none', markerfmt='k.', label='original signal', basefmt=" ")
ax.scatter(x[outliers], y[outliers], color = "red", label = "outliers", s = 70)
ax.set_title(f"Support Vector Regression fit, C = {C}, epsilon = {epsilon}, gamma = {gamma:.2f}")
ax.set_xlabel("Time (t)")
ax.set_ylabel("y(t)")
ax.set_ylim([-0.78, 1.05])

x_ticks = np.append(ax.get_xticks(), SVM_argmax)
ax.set_xticks(x_ticks)
ax.legend()

ax.set_aspect('auto')
ax.set_xlim([-0.2, 12.6])
ax.set_xbound(lower=-0.2, upper=12.8)

plt.show()

   
