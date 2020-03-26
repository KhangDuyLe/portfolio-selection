import numpy as np
import pandas as pd
import parm 
import pickle


def print_hello():
    print("Hello world")


if __name__ == "__main__":
    a = np.random.rand(10)
    b = parm.PARM()
    b.call()
    print(a)
    d = pd.DataFrame({"a":np.ones(5), "b":[i for i in range(5)]})
    path = "./data/nyse_n.pkl"
    print(pd.read_pickle(path))
    print(d)
    print_hello()    
    pass