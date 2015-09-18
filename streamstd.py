import math
import random
import numpy as np
import pandas as pd
random.seed(0)


def update_avg(x, n, prev_mean):
    """
    online update of the avg given previous avg

    x = current sample value to add to rolling avg
    n = index into window
    prev_mean = mean at n-1
    """
    ans = (n*prev_mean + x) / (n+1)
    # print('%s = (%s*%s + %s) / (%s+1)' % (ans, n, prev_mean, x, n))
    return ans


def analogRead():
    return random.random() ** 10


max_samples = 300
n_windows = 150
# validation
if max_samples / n_windows != max_samples // n_windows:
    raise Exception("n_windows should be a factor of max_samples")

n = [x * max_samples / n_windows for x in range(n_windows)]
prev_mean = [0] * n_windows
prev_std_sum = [0] * n_windows


cur_avg = 0
sum_sq_err = 0

xs = []  # debug
data = []  # debug: testing

for ith_iter in range(1000):
    x = analogRead()

    # maintain the estimated rolling average
    for i in range(n_windows):
        prev_mean[i] = update_avg(x, n[i], prev_mean[i])
    # choose the window with most samples as current avg estimate
    prev_avg = cur_avg
    if ith_iter < max_samples:
        cur_avg = prev_mean[0]
    else:
        cur_avg = \
            prev_mean[
                n_windows - 1 - (ith_iter % max_samples)
                // (max_samples // n_windows)]
        # above craziness gets index of window with most samples
    for i in range(n_windows):
        prev_std_sum[i] += (x - prev_avg) * (x - cur_avg)

    xs.append(x)  # debug

    # estimate the var and std
    if ith_iter < max_samples:
        sum_sq_err = prev_std_sum[-1]
    else:
        sum_sq_err = prev_std_sum[
            (ith_iter % max_samples) // (max_samples // n_windows)]

    var = sum_sq_err / (max_samples - 1)
    # print(_tmp, n[_tmp])
    std = math.sqrt(var)

    # debug: true rolling std
    tavg = np.mean(xs[-max_samples:])
    tvar = np.var(xs[-max_samples:])
    tstd = np.std(xs[-max_samples:])

    dct = dict(
        est_var=var, est_std=std, est_avg=cur_avg,
        true_var=tvar, true_std=tstd, true_avg=tavg
    )

    # indexing for std
    if ith_iter % (max_samples // n_windows) == 1:
        prev_std_sum[(ith_iter % max_samples) // (max_samples // n_windows)] = 0

    # indexing for avgs
    for x in range(n_windows):
        n[x] = (n[x] + 1) % max_samples

    data.append(dct)

df = pd.DataFrame(data)
