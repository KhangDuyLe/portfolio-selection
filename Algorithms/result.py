import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# from universal import tools
import seaborn as sns
# from statsmodels.api import OLS
from . import tools

from matplotlib.colors import ListedColormap

class AlgoResult():
    """ Results returned by algo's run method. The class containts useful
    metrics such as sharpe ratio, mean return, drawdowns, ... and also
    many visualizations.
    You can specify transactions by setting AlgoResult.fee. Fee is
    expressed in a percentages as a one-round fee.
    """

    def __init__(self, X, B):
        """
        :param X: Price relatives.
        :param B: Weights.
        """
        # set initial values
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X

        # update logarithms, fees, etc.
        self._recalculate()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        # self._recalculate()
    
    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        # self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        r = (self.X - 1) * self.B
        self.asset_r = r + 1
        self.r = r.sum(axis=1) + 1

        # stock went bankrupt
        self.r[self.r < 0] = 0.  
        self.r_log = np.log(self.r)

    @property
    def total_wealth(self):
        return self.r.prod()

    @property
    def equity(self):
        return self.r.cumprod()
    
    @property
    def asset_equity(self):
        return self.X.cumprod()

    @property
    def sharpe(self):
        """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year.
        """
        return tools.sharpe(self.r_log, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def information(self):
        """ Information ratio benchmarked against uniform CRP portfolio. """
        s = self.X.mean(axis=1)
        x = self.r_log - np.log(s)

        mu, sd = x.mean(), x.std()

        freq = self.freq()
        if sd > 1e-8:
            return mu / sd * np.sqrt(freq)
        elif mu > 1e-8:
            return np.inf * np.sign(mu)
        else:
            return 0.
    
    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()

    @property
    def annualized_return(self):
        return np.exp(self.r_log.mean() * self.freq()) - 1

    @property
    def annualized_volatility(self):    
        return np.exp(self.r_log).std() * np.sqrt(self.freq())

    @property
    def drawdown_period(self):
        ''' Returns longest drawdown perid. Stagnation is a drawdown too. '''
        x = self.equity
        period = [0.] * len(x)
        peak = 0
        for i in range(len(x)):
            # new peak
            if x[i] > peak:
                peak = x[i]
                period[i] = 0
            else:
                period[i] = period[i-1] + 1
        return max(period) * 252. / self.freq()

    @property
    def max_drawdown(self):
        ''' Returns highest drawdown in percentage. '''
        x = self.equity
        return max(1. - x / x.cummax())

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    @property
    def turnover(self):
        return self.B.diff().abs().sum().sum()

    def freq(self, x=None):
        """ Number of data items per year. If data does not contain
        datetime index, assume daily frequency with 252 trading days a year."""
        x = x or self.r
        return tools.freq(x.index)
    
    @property
    def profit_factor(self):
        x = self.r_log
        up = x[x > 0].sum()
        down = -x[x < 0].sum()
        return up / down if down != 0 else np.inf

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()
