import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import os


from Algorithms import pamr
from Algorithms import olmar
from Algorithms import anticor

def test1():
    algos = [
        pamr.PAMR(),
        pamr.PAMR(variant=1),
        pamr.PAMR(variant=2),
    ]

    algo_names = [a.__class__.__name__ for a in algos]
    algo_data = ['algo', 'results', 'profit', 'sharpe', 'information', 'annualized_return', 'drawdown_period','wealth_cumulative','volatility']
    metrics = algo_data[2:]
    print(algo_names)


    dataInfo = {
            "nyse_o":"./data/nyse_o.pkl",
            "nyse_n":"./data/nyse_n.pkl",
            "tse": "./data/tse.pkl",
            "sp500": "./data/sp500.pkl",
            "msci": "./data/msci.pkl",
            "djia": "./data/djia.pkl"
            }

    data_name = [key for key in dataInfo]
    print(data_name)

    data = pd.read_pickle(dataInfo["nyse_o"])

    olps_train = pd.DataFrame(index=algo_names, columns=algo_data)
    olps_train.algo = algos

    for name, alg in zip(olps_train.index, olps_train.algo):
        olps_train.ix[name,'results'] = alg.run(data)

    def olps_stats(df):
        for name, r in df.results.iteritems():
            df.ix[name,'profit'] = r.profit_factor
            df.ix[name,'sharpe'] = r.sharpe
            # df.ix[name,'information'] = r.information
            df.ix[name,'annualized_return'] = r.annualized_return * 100
            df.ix[name,'drawdown_period'] = r.drawdown_period
            
            df.ix[name,'wealth_cumulative'] = r.total_wealth
            df.ix[name,'volatility'] = r.volatility
            # df.ix[name,'winning_pct'] = r.winning_pct * 100
        return df

    olps_stats(olps_train)
    olps_train[metrics].sort_values('profit', ascending=False)
    olps_train.to_csv('res.csv', index=False)

# detect the current working directory and print it
# path = os.getcwd()
# print ("The current working directory is %s" % path)

# path += "/tmp"

# try:
#     os.mkdir(path)
# except OSError:
#     print ("Creation of the directory %s failed" % path)
# else:
#     print ("Successfully created the directory %s " % path)

# try:
#     os.rmdir(path)
# except OSError:
#     print ("Deletion of the directory %s failed" % path)
# else:
#     print ("Successfully deleted the directory %s" % path)

def check_data(file1, file2, shape):
    path = os.getcwd()

    with open(path +file1, "r") as f:
        array = []
        for line in f:
            # temp = [float(x) for x in line.split("   ")]
            # array.append(temp)
            t = line.split("  ")
            for i in range(1, len(t)-1):
                # print(t[i])
                array.append(float(t[i]))
            array.append(float(t[len(t)-1].replace('\n','')) )
    matrix = np.reshape(array, shape)

    data = pd.read_pickle(file2)
    S =data 
    X = S / S.shift(1).fillna(method='ffill')
    X.iloc[0,:] = S.iloc[0,:]

    print(file1,  ((X.to_numpy() - matrix)**2).sum() ) 

data = pd.read_pickle("./data/nyse_o.pkl")
S =data 
X = S / S.shift(1).fillna(method='ffill')
X.iloc[0,:] = S.iloc[0,:]
# print(X.columns)
X.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',  
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', 'cc',
       ']', '3', '1', '2', 'a', 'b', 'c', 'd']
X.plot()
plt.savefig('foo.png')

# check_data("/dataset/NYSE.txt","./data/nyse_o.pkl", (5651,36))
# check_data("/dataset/TSE.txt","./data/tse.pkl", (1259,88))
# check_data("/dataset/SP500.txt","./data/sp500.pkl", (1276,25))
# check_data("/dataset/DJIA.txt","./data/djia.pkl", (507,30))