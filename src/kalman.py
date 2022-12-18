import numpy as np
import matplotlib.pyplot as plt

# Set the parameters

q = 1
dt = 0.1
s = 0.5
F    = np.array([
    [1, dt],
    [0, 1],
])
Q = q*np.array([
    [dt**3/3, dt**2/2],
    [dt**2/2, dt],
])

H = np.array([
    [1, 0],
])

R = s**2
m0 = np.array([[0], [1]])
P0 = np.identity(2)

# Simulate data
def sim(seed, steps):
    np.random.seed(seed+2001)

    X = np.zeros((len(F), steps))
    Y = np.zeros((len(H), steps))
    x = m0

    for k in range(steps):
        q = np.linalg.cholesky(Q)@np.random.randn(len(F), 1)
        x = F@x + q
        y = H@x + s*np.random.randn(1, 1)
        X[:, k] = x[:, 0]
        Y[:, k] = y[:, 0]

    m = m0
    P = P0
    kf_m = np.zeros((len(m), Y.shape[1]))
    kf_P = np.zeros((len(P), P.shape[1], Y.shape[1]))

    for k in range(Y.shape[1]):
        m = F@m
        P = F@P@F.T + Q

        v = Y[:, k].reshape(-1, 1) - H@m
        S = H@P@H.T + R
        K = P@H.T@np.linalg.inv(S)
        m = m + K@v
        P = P - K@S@K.T

        kf_m[:, k] = m[:, 0]
        kf_P[:, :, k] = P

    rmse_raw = np.sqrt(np.mean(np.sum((Y - X[:1, :])**2, 1)))
    rmse_kf = np.sqrt(np.mean(np.sum((kf_m[:1, :] - X[:1, :])**2, 1)))

    return X, Y, kf_m, rmse_raw, rmse_kf

X, Y, kf_m, rmse_raw, rmse_kf = sim(0, 1000)

print("RMSE of raw estimate: ", rmse_raw)
print("RMSE of KF estimate: ", rmse_kf)

plt.figure()
plt.plot(X[0, :], '-')
plt.plot(Y[0, :], 'o')
plt.xlabel("n")
plt.ylabel("x")
# plt.xticks(range(0, 10))
plt.plot(kf_m[0, :], '-')
plt.title("Object moving in 1D")
plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
plt.show()

# Below 1000 sumulations are run and the RMSE of the raw and KF estimates are plotted
def mean_rmse(n, rang = range(1,101,5)):
    means_kf = np.zeros(len(rang))
    means_raw =  np.zeros(len(rang))

    for index,j in enumerate(rang):
        rmse_raw = np.zeros(n)
        rmse_kf = np.zeros(n)

        for i in range(n):
            X, Y, kf_m, rmse_raw[i], rmse_kf[i] = sim(i, j)

        mean_rmse_raw = np.mean(rmse_raw)
        mean_rmse_kf = np.mean(rmse_kf)

        means_kf[index] = mean_rmse_kf
        means_raw[index] = mean_rmse_raw

    return means_kf, means_raw

# means_kf, means_raw = mean_rmse(1000)
# np.save('means_kf.npy', means_kf)
# np.save('means_raw.npy', means_raw)

def variable_plot():
    means_kf = np.load('means_kf.npy')
    means_raw = np.load('means_raw.npy')

    plt.figure()
    plt.plot(range(1,101,5), means_raw, label='Raw')
    plt.plot(range(1,101,5), means_kf, label='KF')
    plt.xlabel('Number of steps/observations')
    plt.ylabel('RMSE')
    plt.legend()
    xticks = np.linspace(10,100,10)
    xticks = np.append(xticks, 1)
    xticks = np.append(xticks, 100)
    plt.xticks(xticks)
    plt.title('RMSE of raw observations and KF estimates')

    plt.show()


# plt.figure()
# plt.hist(rmse_raw, bins=20, alpha=0.5, label='Raw')
# plt.hist(rmse_kf, bins=20, alpha=0.5, label='KF')
# plt.xlabel('RMSE')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title('RMSE of raw and KF estimates')

# plt.show()

