import pandas as pd
import random
from smart_brake_light import streamstd as stats
import matplotlib.pyplot as pylab

import numpy as np
import scipy.stats as st

pylab.ion()

c = 0
a = 400
braking = False


def analogRead():
    global c, braking, a
    c += 1

    if random.random() > .999:
        # correct towards middle
        a += (random.random() * (int(random.random() >= (a / 800)) * 2 - 1))
    else:
        a += (random.random() * (int(random.random() >= .5) * 2 - 1))
    if a < 0:
        a = 0
    if a > 800:
        a = 800
    if braking:
        braking = random.random() < .98
    else:
        braking = random.random() < .002
    if braking:
        #  return 30
        return a * (.41 + np.abs(random.random()))
    else:
        return a


def main(max_samples, n_windows, lookback):
    ssmean = stats.StreamStats(max_samples, n_windows)
    ssstd = stats.StreamStats(max_samples * 10, n_windows)

    data = []  # debug: testing

    p_z_gt_1 = (1 - st.norm.cdf(1))

    for ith_iter in range(5000):
        x = analogRead()

        cur_avg = stats.update_means(x, ith_iter, ssmean)
        # TODO: std bug fix:  why does multiply by n>1 make more accurate?
        std = stats.update_std(x, ith_iter, ssstd)
        # TEST  -- using true std
        #  std = np.std([_['x'] for _ in data])
        # TEST  -- using rolling std
        #  std = np.std(
            #  [_['x'] for _ in data[-max_samples*3:]] if data else [0])
        z = (x - cur_avg) / std if std != 0 else 0
        ratio = (z > 1) / (p_z_gt_1 * lookback)

        # debug / plot
        dct = dict(
            x=x, o=std, u=cur_avg,
            z=z,

            ratio=ratio,

            # DEBUG
            t_rolling_mean=1,
            #  t_rolling_std=1,
            tavg=1,
            #  tstd=1,
            #  t_rolling_mean=np.mean(
            #      [_['x'] for _ in data[-max_samples:]] if data else [0]),
            t_rolling_std=np.std(
                [_['x'] for _ in data[-max_samples*10:]] if data else [0]),
            #  tavg=np.mean([_['x'] for _ in data]),
            tstd=np.std([_['x'] for _ in data]),
        )
        data.append(dct)  # debug
    return pd.DataFrame(data), ssmean, ssstd


if __name__ == '__main__':
    MAX_SAMPLES = 100
    N_WINDOWS = 20
    LOOKBACK = MAX_SAMPLES / N_WINDOWS

    df, ssmean, ssstd = main(MAX_SAMPLES, N_WINDOWS, LOOKBACK)

    df['z1'] = pd.rolling_sum(df['ratio'], LOOKBACK)

    # experimenting
    #  Z_THRESH = [.5,1, 2, 3,4,5]
    #  for thresh in Z_THRESH:
    #      prob_zscore = st.norm.cdf(thresh)
    #      df['z%s' % thresh] = pd.rolling_sum(
    #          (df['z'] > thresh) / (prob_zscore * LOOKBACK),
    #          LOOKBACK)

    # plotting
    #  pylab.figure()
    #  df[['z%s' % thresh for thresh in Z_THRESH]]\
    #      .plot(subplots=True, ylim=(0, 2))
    #  pylab.figure()
    #  df[['z%s' % thresh for thresh in Z_THRESH]].sum(1).plot()

    # the main plot that shows performance
    f = pylab.figure()
    df['x'].plot(style='g.', alpha=1, legend=True)
    df['u'].plot(style='b', legend=True)
    (df['u'] + df['o']).plot(style='b--')
    (df['u'] - df['o']).plot(style='b--')

    (df['z1'] > 1).plot(secondary_y=('z1', ), color='red')
    df['z1'].plot(secondary_y=('z1', ), style='r.', alpha=.3, legend=True)
    #  df['ratio'].plot(secondary_y=('ratio', ), style='r^', legend=True)



    #  pylab.figure()
    #  df['x'].plot(style='.', color='green', alpha=.3, legend=True)
    #  df['u'].plot(style='b', legend=True)

    # sanity checks
    #  df[['u', 't_rolling_mean', 'tavg']].plot()
    df[['o', 't_rolling_std', 'tstd']].plot()

    #  df[['x', 'u', 'o', 'z']].plot(subplots=True, gid=3, sharey=True)

    # idea: estimate z_thresh based on the most likely % of times we're braking
    # in a given sample period.
    # ie p(braking) = .012 == num_brakin / # num_samples


    # fyi:  on arduino, max samples I could theoretically store before int
    # overflow on summation would be:
    # (1<<(32-1)) / 1024 --> >2M
