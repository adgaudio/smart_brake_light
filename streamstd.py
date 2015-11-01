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
random.seed(0)


def update_means(x, ith_iter, ss):
    """
    Return the mean of the window with the most samples.
    This is the best mean across all windows

    This function modifies state of `ss`

    x = current sample value to add to rolling avg
    ss = StreamStats instance
    """
    # update windows
    for i in range(ss.n_windows):
        n = (ith_iter + i * ss.max_samples / ss.n_windows) % ss.max_samples
        ss.means[i] = (n * ss.means[i] + x) / (n + 1)

    # get avg from best window
    ss.prev_avg = ss.cur_avg
    if ith_iter < ss.max_samples:
        ss.cur_avg = ss.means[0]
    else:
        ss.cur_avg = \
            ss.means[
                ss.n_windows - 1 - (ith_iter % ss.max_samples)
                // (ss.max_samples // ss.n_windows)]
        # above craziness gets index of window with most samples
    return ss.cur_avg


def update_std_sums(x, ith_iter, ss):
    """
    Calculate an intermediary value for standard deviation for each window
    Return current standard deviation from window with most samples

    x = current sample value to add to rolling avg
    prev_avg = previous best average across all windows
    cur_avg = current best average across all windows
    ss = StreamStats instance
    """
    # update windows
    for i in range(ss.n_windows):
        ss.std_sums[i] += (x - ss.prev_avg) * (x - ss.cur_avg)

    # get std from best window
    sum_sq_err = ss.std_sums[
        (ith_iter % ss.max_samples) // (ss.max_samples // ss.n_windows)]
    if 1 < ith_iter < ss.max_samples:
        # extrapolate what the sum_sq_err should be if we haven't
        # generated max_samples yet
        sum_sq_err = sum_sq_err * ss.max_samples / ith_iter
    var = sum_sq_err / (ss.max_samples - 1)
    std = math.sqrt(var)

    # update indexing
    if ith_iter % (ss.max_samples // ss.n_windows) == \
            ss.max_samples // ss.n_windows - 1:
        ss.std_sums[
            (ith_iter % ss.max_samples) // (ss.max_samples // ss.n_windows)] = 0

    return std


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
        self.means = [0] * n_windows
        self.std_sums = [0] * n_windows

        self.cur_avg = 0
        self.prev_avg = 0


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
    df = main(400)
    df[['est_std', 'true_std']].plot()
    df[['est_avg', 'true_avg']].plot()
