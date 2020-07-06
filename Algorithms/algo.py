import numpy as np
import pandas as pd
import inspect

from .result import AlgoResult

class Algo(object):
    REPLACE_MISSING = False
    PRICE_TYPE = 'ratio'

    def init_step(self, X):
        """ Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """
        pass

    def __init__(self, frequency=1):
        self.name = "main algorithm"
        self.frequency = frequency

    def init_weights(self, m):
        return np.zeros(m)

    def nomarlization(self, data):
        S = data/data.shift(1).fillna(method='ffill')
        S.drop(0, axis=0, inplace=True)
        return S

    def step(self, x, last_b):
         """ Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        """
        # raise NotImplementedError('Subclass must implement this!')

    def weights(self, X, min_history=None, log_progress=True):
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        # min_history = self.min_history if min_history is None else min_history

        # init
        B = X.copy() * 0.
        last_b = self.init_weights(X.shape[1])
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        # use history in step method?
        use_history = self._use_history_step()

        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.ix[t] = last_b

            # keep initial weights for min_history
            # if t < min_history:
            #     continue

            # trade each `frequency` periods
            # if (t + 1) % self.frequency != 0:
            #     continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t+1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            # if log_progress:
            #     tools.log_progress(t, len(X), by=10)

        return B

    @classmethod
    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == 'raw':
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.iteritems():
                init_val = s.ix[s.first_valid_index()]
                r[name] = s / init_val
            X = pd.DataFrame(r)

            if replace_missing:
                X.ix[0] = 1.
                X = X.fillna(method='ffill')

            return X

        elif method == 'absolute':
            return S

        elif method in ('ratio', 'log'):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method='ffill')
            for name, s in X.iteritems():
                X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.

            if replace_missing:
                X = X.fillna(1.)
            
            return np.log(X) if method == 'log' else X

        else:
            raise ValueError('invalid price conversion method')

    def _use_history_step(self):
        """ Use history parameter in step method? """
        step_args = inspect.getargspec(self.step)[0]
        return len(step_args) >= 4


    def run(self, S):
        # S = self.nomarlization(data)
        # n, m = S.shape
        X = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        B = self.weights(X)

        return AlgoResult(X , B)