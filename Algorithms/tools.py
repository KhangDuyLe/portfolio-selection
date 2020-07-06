import numpy as np
import pandas as pd

    
def simplex_proj(y):
    """ Projection of y onto simplex. """
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] -1)/m

    return np.maximum(y-tmax,0.)

def sharpe(r_log, rf_rate=0., alpha=0., freq=None, sd_factor=1.):
    """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year

        TODO: calculate real sharpe ratio (using price relatives), see
            http://www.treasury.govt.nz/publications/research-policy/wp/2003/03-28/twp03-28.pdf
    """
    freq = freq or _freq(r_log.index)

    mu, sd = r_log.mean(), r_log.std()

    # annualize return and sd
    mu = mu * freq
    sd = sd * np.sqrt(freq)

    # risk-free rate
    rf = np.log(1 + rf_rate)

    sh = (mu - rf) / (sd + alpha)**sd_factor

    if isinstance(sh, float):
        if sh == np.inf:
            return np.inf * np.sign(mu - rf**(1./freq))
    else:
        sh[sh == np.inf] *= np.sign(mu - rf**(1./freq))
    return sh

def freq(ix):
    """ Number of data items per year. If data does not contain
    datetime index, assume daily frequency with 252 trading days a year."""
    assert isinstance(ix, pd.Index), 'freq method only accepts pd.Index object'

    # sort if data is not monotonic
    if not ix.is_monotonic:
        ix = ix.sort_values()

    if isinstance(ix, pd.DatetimeIndex):
        days = (ix[-1] - ix[0]).days
        return len(ix) / float(days) * 365.
    else:
        return 252.