import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def setup(results_dir):
    """Determine the data structure.

    :param results_dir: The directory where the results are located.
    :return: An empty pandas dataframe with the columns filled in, and the expected number of samples for
    each set of data.
    """
    n_trials = 0
    df = None

    for path, subdirs, files in os.walk(results_dir):
        if not subdirs:  # i.e. leaf directory
            with open(path + '/statistics.json', 'r') as f:
                stats = json.load(f)

            metrics = set()

            for filename in files:
                if filename.endswith('.npy'):
                    filepath = '%s/%s' % (path, filename)
                    a = np.load(filepath)

                    if n_trials > 0:
                        assert a.shape[0] == n_trials, 'Inconsistent number of trials in data. Expected %d ' \
                                                       'trials, but instead got %d from the file \'%s\'' \
                                                       % (n_trials, a.shape[0], filepath)
                    else:
                        n_trials = a.shape[0]

                    metrics.add(filename[:-4])

            param_names = list(stats['params'].keys())
            data_fields = ['%s_%02d' % (metric, n) for metric in metrics for n in range(n_trials)]
            df = pd.DataFrame(columns=['run_id'] + param_names + data_fields)
            df = df.set_index('run_id')

            break

    return df, n_trials


def main(results_dir, output_file):
    df, n_trials = setup(results_dir)
    run_number = 0

    start = datetime.now()

    for path, subdirs, files in os.walk(results_dir):
        if not subdirs:  # i.e. leaf directory
            with open(path + '/statistics.json', 'r') as f:
                stats = json.load(f)

            run_number += 1
            print('\rProcessing run %d (%s) - Elapsed Time: %s' % (run_number, stats['run_id'], datetime.now() - start),
                  end='')

            series = pd.Series(stats['params'], index=df.columns)

            for filename in files:
                if filename.endswith('.npy'):
                    metric = filename[:-4]
                    filepath = '%s/%s' % (path, filename)
                    a = np.load(filepath)

                    assert a.shape[0] == n_trials, 'Expected %d samples, but the data from the file \'%s\' ' \
                                                   'implies only %d samples.' % (n_trials, filepath, a.shape[0])

                    a = a.T
                    a = pd.DataFrame(a)
                    a = a.replace([np.inf, -np.inf], np.nan)

                    # Find the last valid data point (highest epoch number).
                    idx = a.apply(pd.Series.last_valid_index)

                    for col, row in enumerate(idx):
                        # row can be None (`a` was all NaNs or infs) or
                        # nan (this row in `a` was all NaNs or infs).
                        if row and not np.isnan(row):
                            the_column = '%s_%02d' % (metric, col)
                            # `row` is sometimes a float, indexing requires ints
                            series[the_column] = a.iloc[int(row), col]

            df.loc[stats['run_id']] = series

    print()
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a compressed view of the results.\n'
                                                 'A pandas dataframe is generated which for each run (or grid search '
                                                 'configuration) the hyperparameter settings, and the last valid value '
                                                 'for the training loss, training scores, validation loss, and '
                                                 'validation scores.')
    parser.add_argument('-r', '--results-dir', type=str, default='../results', help='Where the results are located.')
    parser.add_argument('-o', '--output', type=str, default='results_summary.csv',
                        help='The file to save the generated dataframe to.')

    args = parser.parse_args()

    main(args.results_dir, args.output)
