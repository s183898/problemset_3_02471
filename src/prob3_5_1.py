import numpy as np
import matplotlib.pyplot as plt

# Set the parameters

q = 1
dt = 0.1
s = 0.2
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
    np.random.seed(seed)

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
    
    X_real = X[0, :]
    X_kf = kf_m[0, :]
    X_obs = Y[0, :]

    rmse_raw = np.sqrt(np.mean((X_real - X_obs)**2))
    rmse_kf = np.sqrt(np.mean((X_real - X_kf)**2))


    return X, Y, kf_m, rmse_raw, rmse_kf


def plot_single_rollout(seed, steps):
    X, Y, kf_m, rmse_raw, rmse_kf = sim(seed, steps)

    print("RMSE of raw estimate: ", rmse_raw)
    print("RMSE of KF estimate: ", rmse_kf)

    plt.figure()
    plt.plot(X[0, :], '-')
    plt.plot(Y[0, :], 'o')
    plt.xlabel("n")
    plt.ylabel("p (position)")
    # plt.xticks(range(0, 10))
    plt.plot(kf_m[0, :], '-')
    plt.title("Object moving in 1D, sigma = 0.2, q = 1, dt = 0.1")
    plt.legend(['True Trajectory', 'Measurements', 'Filter Estimate'])
    plt.show()

plot_single_rollout(0, 10)


def mean_rmse(n, rang = range(1,101,5)):
    means_kf = np.zeros(n)
    means_raw =  np.zeros(n)

    for index,j in enumerate(rang):
        rmse_raw = np.zeros(n)
        rmse_kf = np.zeros(n)

        for i in n:
            _, _, _, rmse_raw[i], rmse_kf[i] = sim(n, j)

        mean_rmse_raw = np.mean(rmse_raw)
        mean_rmse_kf = np.mean(rmse_kf)

        means_kf[index] = mean_rmse_kf
        means_raw[index] = mean_rmse_raw

    return means_kf, means_raw

# np.save('means_kf.npy', means_kf)
# np.save('means_raw.npy', means_raw)

def variable_step_plot(n, rang = range(1,101,5)):

    means_kf, means_raw = mean_rmse(n, rang)

    plt.figure()
    plt.plot(rang, means_raw, label='Observations')
    plt.plot(rang, means_kf, label='KF estimate')
    plt.xlabel('Number of steps/observations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.xticks([1,5,10,20,30,40,50,60,70,80,90,100])
    plt.title('RMSE of observations and KF estimates')
    plt.show()


def single_rollout_hist(rmse_raw, rmse_kf):
    plt.figure()
    plt.hist(rmse_raw, bins=20, alpha=0.5, label='Raw')
    plt.hist(rmse_kf, bins=20, alpha=0.5, label='KF')
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('RMSE of raw and KF estimates')
    plt.show()

