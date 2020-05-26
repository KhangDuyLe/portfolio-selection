import numpy as np 
import pandas as pd

import time 

import scipy.optimize as opt


def run(x):
    '''
    Best stock strategy which is a BAH strategy that put all capital on the stock with the best performance in hindsight.
    Input: 
        x : relative price has size of (n,m)
        n : number of periods (day, month or week)
        m : number of assets
        Output: cumulative wealth
    '''

    n, m = x.shape

    b = np.ones((1,m))/m
    bb = [(0,1) for i in range(m)]
    bb =tuple(bb)

    def objective(b):
        temp = np.ones((m,1))
        S0= 1
        for t in range(n):
            temp = temp *x.iloc[t,:].values.reshape(m,1)
        S0 = S0*np.matmul(b,temp)
        # print(S0)
        return -S0[0]
    
    cons1 = {'type':'eq','fun': lambda b: np.sum(b)-1}
    cons = [cons1]

    return opt.minimize(objective, b.flatten(), bounds=bb, method= 'SLSQP', constraints=cons )