"""
Builds an online algorithm for Standard Deviation and Approximate Rolling
Average.
"""
from __future__ import division
import math
import random
import numpy as np
import pandas as pd
import pylab
pylab.ion()


def update_means(x, ith_iter, ss):
    """
    Return the mean of the window with the most samples.
    This is the best mean across all windows

    This function modifies state of `ss`

    x = current sample value to add to rolling avg
    ss = StreamStats instance

    O(n_windows + c)
    """
    ith_iter = ith_iter % ss.max_samples  # hack fix
    _window_size = ss.max_samples / ss.n_windows

    # get index of window with least elements in it
    i = int(ith_iter // _window_size % ss.n_windows)
    # get num elements that make up the sum (including value about to add)
    n = (ith_iter % _window_size) + 1

    # update sum for window with least elements in it
    if n == 1:
        ss.sums[i] = x
    else:
        ss.sums[i] += x

    # calculate estimated rolling average from windows
    if n == _window_size:  # basecase, all sums are full
        avg = sum(ss.sums[x] / ss.max_samples for x in range(ss.n_windows))
    else:
        # rather than take a strict rolling avg, benefit from any extra data
        avg = sum(ss.sums[x] / (n + _window_size * (ss.n_windows - 1))
                  for x in range(ss.n_windows))
        # sanity check
        assert n + _window_size * (ss.n_windows - 1) < ss.max_samples

    return avg


def update_std(x, ith_iter, ss):
    """
    Online Standard Deviation algorithm

    x = current sample value to add to rolling avg
    ss = StreamStats instance
    """
    ms = ss.max_samples  # max samples for std calculation

    ith_iter = ith_iter % ms  # hack fix

    if ss.prev_avg is None:  # basecase
        ss.prev_avg = x
        return 1

    # get num elements that make up the sum (including value about to add)
    delta = (x - ss.prev_avg)
    cur_avg = ss.prev_avg + delta / ms
    # --> TODO first time around, denominator is ith_iter.  maybe leave as is

    # update vars
    ss.prev_avg = cur_avg
    err = delta * (x - cur_avg)
    # update hack: store multiple sum sq errs and reset them every so often
    for i in range(len(ss.sum_sq_err)):
        ss.sum_sq_err[i] += err

    # get index of oldest sum_sq_err
    _i = int(ith_iter / (ms / len(ss.sum_sq_err)))
    # re-initialize youngest sum_sq_err to 0
    if ith_iter % (ms / len(ss.sum_sq_err)) == 0:
        ss.sum_sq_err[_i - 1] = 0

    return np.sqrt(ss.sum_sq_err[_i] / (ms - 1))
    # TODO: estimage sqrt?


def analogRead():
    return random.random() ** 10


class StreamStats(object):
    """Store data to calculate rolling average and standard deviation
    using low memory
    """
    def __init__(self, max_samples, n_windows):
        """
        `max_samples` how many data points, or how much history, to consider
        `n_windows` how accurate the estimates of the avg and stdev should be.
            the tradeoff here is more windows == more ram
        """
        # validation
        if max_samples / n_windows != max_samples // n_windows:
            raise Exception("n_windows should be a factor of max_samples")

        self.max_samples = max_samples
        self.n_windows = n_windows

        # other variables
        self.sums = [0] * n_windows
        self.std_sums = [0] * n_windows

        self.prev_avg = None
        self.sum_sq_err = [0] * n_windows  # TODO: make this configurable


def main(max_samples, n_windows=None):
    if n_windows is None:
        # guess acceptable window size
        n_windows = int(2*math.log(max_samples))
    ss = StreamStats(max_samples, n_windows)

    xs = []  # debug
    data = []  # debug: testing

    for ith_iter in range(1000):
        x = analogRead()

        # maintain the estimated rolling average
        cur_avg = update_means(x, ith_iter, ss)
        std = update_std_sums(x, ith_iter, ss)

        # debug: true rolling std
        xs.append(x)  # debug
        tavg = np.mean(xs[-max_samples:])
        tstd = np.std(xs[-max_samples:])

        dct = dict(
            est_std=std, est_avg=cur_avg,
            true_std=tstd, true_avg=tavg
        )

        data.append(dct)
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    pass
    #  random.seed(0)
    #  df = main(400)
    #  df[['est_std', 'true_std']].plot()
    #  df[['est_avg', 'true_avg']].plot()
