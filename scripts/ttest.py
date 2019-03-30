import argparse

import numpy as np
import pandas as pd
from scipy import stats


def _int2char(integer):
    if integer > 26:
        raise NotImplementedError('int2char not implemented for integers greater than 26.')

    # 97 is the offset of the ASCII character 'a'.
    return chr(integer + 97)


def ttest(cfgs, metric='val_scores', greater_than=True, results_dir='../results'):
    config_data = []
    ttest_results = None

    print('%s distribution per configuration (μ ± 2σ)' % metric)

    for cfg in cfgs:
        a = np.load(results_dir + '/%s/%s.npy' % (cfg, metric))
        a = a.T
        a = pd.DataFrame(a)
        a = a.replace([np.inf, -np.inf], np.nan)

        # Find the last valid data point (highest epoch number).
        idx = a.apply(pd.Series.last_valid_index)
        idx = idx.dropna()

        if len(idx) > 0:
            idx = idx.astype(np.int)
            a = a.values[idx, idx.index]
            print('%s: %.4f ± %.4f (n=%d)' % (_int2char(len(config_data)), a.mean(), 2 * a.std(), len(a)))
        else:
            a = np.array([])
            print('%s: NaN' % _int2char(len(config_data)))

        config_data.append(a)

    for i in range(len(config_data)):
        baseline = config_data[i]

        for j in range(i + 1, len(config_data)):
            if greater_than:
                d = '>'
                ttest_results = stats.ttest_ind(config_data[j], baseline, equal_var=False)
            else:
                d = '<'
                # The hypothesis is that x < baseline, or baseline > x. Therefore baseline is placed on lhs since we
                # expect it to be larger. This means a positive t-statistic can be interpreted as positive support for
                # the hypothesis and vice versa.
                ttest_results = stats.ttest_ind(baseline, config_data[j], equal_var=False)

            print('%s %s %s:' % (_int2char(j), d, _int2char(i)), ttest_results)

    print()

    return ttest_results


def main(params, results_dir):
    try:
        free_variable = next(key for key in params if type(params[key]) is list)
    except StopIteration:
        print('One parameter must be a list.')
        raise

    print('Parameters: %s' % params)
    for i, value in enumerate(params[free_variable]):
        print('%s: %s=%s' % (_int2char(i), free_variable, str(value)))
    print()

    cfgs = []

    for i in range(len(params[free_variable])):
        kv_pairs = []

        for key, value in zip(params.keys(), params.values()):
            if key == free_variable:
                value = value[i]

            # This is necessary since when a hyper parameter is zero it is printed as an integer, however
            # zero-valued float print as '0.0'.
            if type(value) is float and value == 0:
                value = int(value)

            kv_pairs.append('%s=%s' % (key, str(value)))

        cfg = '/'.join(kv_pairs)
        cfgs.append(cfg)

    return (ttest(cfgs, 'train_loss', greater_than=False, results_dir=results_dir),
            ttest(cfgs, 'train_scores', results_dir=results_dir),
            ttest(cfgs, 'val_loss', greater_than=False, results_dir=results_dir),
            ttest(cfgs, 'val_scores', results_dir=results_dir))


def _unwrap(a):
    """Unwrap lists of single values.

    Arguments:
        a: The list to to unwrap.

    Returns: The first element of `a` if `a` only contains a single value, otherwise `a`.
    """
    return a[0] if len(a) == 1 else a


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Welch t-test on a subset of the experiment results.\n'
                                                 'One parameter should be defined as a list of values. For example, '
                                                 'if you wanted to see what batch size gives better results then you '
                                                 'could set the batch size arguments to `[16, 32]`')
    parser.add_argument('-r', '--results-dir', type=str, default='../results', help='Where the results are located.')
    parser.add_argument('-a', '--activation-func', type=str, nargs='+', required=True)
    parser.add_argument('-b', '--batch-size', type=int, nargs='+', required=True)
    parser.add_argument('-c', '--clf-type', type=str, nargs='+', required=True)
    parser.add_argument('-d', '--dataset', type=str, nargs='+', required=True)
    parser.add_argument('-g', '--gaussian-noise', type=float, nargs='+', required=True)
    parser.add_argument('-l', '--learning-rate', type=float, nargs='+', required=True)
    parser.add_argument('-m', '--momentum', type=float, nargs='+', required=True)
    parser.add_argument('-s', '--shuffle-batches', type=lambda v: type(v) is str and v.lower() == 'true',
                        nargs='+', required=True)

    args = parser.parse_args()

    params = {
        'activation_func': _unwrap(args.activation_func),
        'batch_size': _unwrap(args.batch_size),
        'clf_type': _unwrap(args.clf_type),
        'dataset': _unwrap(args.dataset),
        'gaussian_noise': _unwrap(args.gaussian_noise),
        'learning_rate': _unwrap(args.learning_rate),
        'momentum': _unwrap(args.momentum),
        'shuffle_batches': _unwrap(args.shuffle_batches)
    }

    main(params, args.results_dir)
