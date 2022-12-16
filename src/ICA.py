import numpy as np

# Independent Component Analysis function
def ICA(x, mu, num_components, iters, mode):
    # random initialization
    W = np.random.rand(num_components, num_components)
    N = np.size(x, 0)

    if mode=='superGauss':
        phi = lambda u : 2*np.tanh(u)
    elif mode=='subGauss':
        phi = lambda u : u-np.tanh(u)
    else:
        print("Unknown mode")
        return W

    for i in range(iters):
        z = W @ x.T   
        dW = (np.eye(num_components) - phi(z) @ z.T/N) @ W    
        # update
        W = W + mu*dW
    return(W)
