"""
Testing methods for calculating avg and std with very low ram usage
for embedded devices
"""
import random
import pandas as pd
from pylab import ion, figure
ion()


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


def approx_rolling_avg(max_samples=40, n_windows=5):
    """
    A very low memory rolling average with configurable accuracy
    Meant for embedded systems

    This is an example function that "samples" data from the analogRead function
    1000 times

    # how far back into history do you want the rolling avg to look?
    # limited by "size of max int" > "max sampled value" * "max sample size"
    max_samples = 40

    # how accurate do you want this to be?  tradeoff: use more ram
    n_windows = 5  # n_windows must be a factor of max_samples
    """
    # initialize vars
    ith_iter = 0
    diffs = []
    xs = []
    n = [x * max_samples / n_windows for x in range(n_windows)]
    prev_mean = [0] * n_windows
    if max_samples / n_windows != max_samples // n_windows:
        raise Exception("n_windows should be a factor of max_samples")

    # void loop()
    # while 1:
    for _ in range(1000):
        x = analogRead()
        for i in range(n_windows):
            prev_mean[i] = update_avg(x, n[i], prev_mean[i])
        xs.append(x)
        if len(xs) <= 1:
            continue
        # collect data to compare est_rollavg against true rolling avg
        dct = {
            'rm': sum(xs) / len(xs),
            'true_rollavg': sum(xs[-max_samples:]) / len(xs[-max_samples:]),
            'iter': ith_iter,
            'val': x,
        }

        # choose which rolling_avg window to use
        if ith_iter < max_samples:
            dct['est_rollavg'] = prev_mean[0]
        else:
            dct['est_rollavg'] = \
                prev_mean[n_windows - 1 - (ith_iter % max_samples)
                          // (max_samples // n_windows)]
            # above craziness gets index of window with most samples

        # all data points used to estimate the rolling avg
        for x in range(n_windows):
            dct.update({
                'est_rollavg_%s' % x: prev_mean[x],
                'idx_window_%s' % x: n[x],
            })
        diffs.append(dct)

        # manage indexes
        ith_iter += 1
        for x in range(n_windows):
            n[x] = (n[x] + 1) % max_samples
    return pd.DataFrame(diffs)


def verify_avg_method_is_working(df, max_samples, n_windows, plot=False):
    min_idx = max_samples / n_windows * (n_windows - 1)
    print('min_idx', min_idx)

    sum_sq_err = ((df['true_rollavg'] - df['est_rollavg']) ** 2).sum()
    print("sum squared error:", sum_sq_err)

    # plotting...
    if plot:

        df[['est_rollavg', 'true_rollavg']].plot(title='%s. avgs' % n_windows)
        figure()
        (df['true_rollavg'] - df['est_rollavg']).plot(
            title='%s. diff between est_rolling_avg and true_rolling_avg'
            % n_windows)

    # more plotting...
    if plot:
        # compare performance of estimate to true rolling avg
        z = pd.DataFrame(
            df[['rm', 'rm']].values
            - df[['est_rollavg', 'true_rollavg']].values,
            columns=['est_rollavg_err', 'true_rollavg_err'])
        z.plot(title='%s. difference from true mean' % n_windows)

    return sum_sq_err


if __name__ == '__main__':
    verify_avg_method_is_working(
        approx_rolling_avg(200, 10),
        200, 10, plot=True)

    # examine sum sq error across all possible windows
    max_samples = 200
    errs = {}
    dfs = []
    for n_windows in range(1, max_samples+1):
        if max_samples // n_windows != max_samples / n_windows:
            continue  # only factors of max_samples should be used
        random.seed(4)
        df = approx_rolling_avg(max_samples, n_windows)
        dfs.append(df)
        errs[n_windows] = verify_avg_method_is_working(
            df, max_samples, n_windows)
    errs = pd.Series(errs)
    # errs.cumsum().plot()
    print(errs)
