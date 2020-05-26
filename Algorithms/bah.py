import pandas as pd 
import numpy as np
import time 

def run(x):
    '''
    This is the most baseline strategy
    Input: 
        x : relative price has size of (n,m)
        n : number of periods (day, month or week)
        m : number of assets
        Output: cumulative wealth
    '''
    n ,m  = x.shape
    # Initialize portfolio : (1xm)
    b = np.ones((1,m))/m
    temp = np.ones((m,1))
    # initialize S0 
    S0 = 1
    start = time.time()
    
    for t in range(n):
        temp = temp*x.iloc[t,:].values.reshape(m,1)
    end = time.time()
    # print("Time: ",end- start)
    S0 = S0*np.matmul(b,temp)

    return S0[0][0]   