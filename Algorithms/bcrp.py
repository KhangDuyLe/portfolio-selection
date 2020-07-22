import pandas as pd 
import numpy as np
import scipy.optimize as optimize


def run(X):
    x_0 = 1 * np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    cons = ({'type': 'eq', 'fun': lambda b: 1 - sum(b)},)
    while True:
        # problem optimization
        res = optimize.minimize(objective, x_0, bounds=[(0., 1)]*len(x_0), constraints=cons, method='slsqp')

        # result can be out-of-bounds -> try it again
        EPS = 1E-7
        if (res.x < 0. - EPS).any() or (res.x > 1 + EPS).any():
            X = X + np.random.randn(1)[0] * 1E-5
            logging.debug('Optimal weights not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution does not exist, use zero weights.')
                res.x = np.zeros(X.shape[1])
            else:
                logging.warning('Converged, but not successfully.')
            break
    return (X*res.x).sum(axis=1).prod()