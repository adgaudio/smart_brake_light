import pandas as pd
import random
from smart_bike_light import streamstd as stats
import pylab
pylab.ion()

import numpy as np
import scipy.stats as st


c = 0
a = 0
braking = False
def analogRead():
    global c, braking, a
    c += 1
    a += (random.random() * ((random.random() >= .5)*2-1))
    if braking:
        braking = random.random() < .98
    else:
        braking = random.random() < .0003
    if braking:
        return np.abs(a * (2 + random.random()))
    else:
        return a


def main(max_samples, n_windows):
    ss = stats.StreamStats(max_samples, n_windows)

    data = []  # debug: testing

    for ith_iter in range(10000):
        x = analogRead()

        if ith_iter % 100 < 10:
            x = x + .1

        cur_avg = stats.update_means(x, ith_iter, ss)
        std = stats.update_std_sums(x, ith_iter, ss)
        z = (x - cur_avg) / std if std != 0 else 0

        dct = dict(
            # debug
            x=x, o=std, u=cur_avg,
            z=z,

            tavg=np.mean(
                [_['x'] for _ in data[-max_samples:]] if data else [0]),
            tstd=np.std([_['x'] for _ in data[-max_samples:]] if data else [1]),
        )
        data.append(dct)  # debug
    return pd.DataFrame(data), ss


if __name__ == '__main__':
    MAX_SAMPLES = 200
    N_WINDOWS = 10
    LOOKBACK = 10  # duration of breaking event when it occurs, in num samples

    # this is like the sensitivity.  closer to 0 means always trigger braking
    # events.  4 means only brake if 4 std deviations away.
    Z_THRESH = [1, 1.5, 2, 2.2, 2.5, 3, 3.5, 4]
    # idea: if I take a smattering of z scores, then I can estimate the most
    # likely zscore threshold.
    # >> maybe just the points that have more than half agreement across
    # thresholds (if num thresholds considered is somehow evaluated)
    # >> probability of z_thresh == sensitivity = ??

    df, ss = main(MAX_SAMPLES, N_WINDOWS)
    for thresh in Z_THRESH:
        prob_zscore = st.norm.cdf(
            thresh, loc=df['u'].values, scale=df['o'].values)
        df['z%s' % thresh] = pd.rolling_sum(
            (df['z'] > thresh) / (prob_zscore) * ss.max_samples,
            LOOKBACK
        ) > 1
    df[['z%s' % thresh for thresh in Z_THRESH] + ['u', 'o', 'x', 'z']]\
        .plot(subplots=True)

    # more windows, less likely to trigger
    # more samples, unclear... nothing
    # lookback, more likely to trigger

    # idea: estimate z_thresh based on the most likely % of times we're braking
    # in a given sample period.
    # ie p(braking) = .012 == num_brakin / # num_samples
