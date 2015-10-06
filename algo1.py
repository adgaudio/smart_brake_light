import pandas as pd
import random
from bike_light_arduino import streamstd as stats
import pylab
pylab.ion()

import numpy as np


def analogRead():
    return random.random()


def main(max_samples, n_windows):
    ss = stats.StreamStats(max_samples, n_windows)

    data = []  # debug: testing

    for ith_iter in range(1000):
        x = analogRead()

        if ith_iter % 100 < 10:
            x = x + .1

        cur_avg = stats.update_means(x, ith_iter, ss)
        std = stats.update_std_sums(x, ith_iter, ss)

        dct = dict(  # debug
            x=x, o=std, u=cur_avg,
            z=(x - cur_avg) / std if std != 0 else 0,

            tavg=np.mean(
                [_['x'] for _ in data[-max_samples:]] if data else [0]),
            tstd=np.std([_['x'] for _ in data[-max_samples:]] if data else [1]),
        )
        data.append(dct)  # debug
    return pd.DataFrame(data), ss


if __name__ == '__main__':
    MAX_SAMPLES = 200
    N_WINDOWS = 20
    # LOOKBACK = 10  # duration of breaking event when it occurs, in num samples

    df, ss = main(MAX_SAMPLES, N_WINDOWS)
    for lb in range(1, 30, 5):
        df['z%s' % lb] = pd.rolling_sum(
            (df['z'] > 2) / .0455 * ss.max_samples, lb) > 1
    df[['z%s' % lb for lb in range(1, 30, 5)] + ['u', 'x']].plot(subplots=True)

    # more windows, less likely to trigger
    # more samples, unclear... nothing
    # lookback, more likely to trigger
