import argparse

import numpy as np
import pandas as pd
from scipy import stats


def main(data_path, output_path):
    df = pd.read_csv(data_path)

    # Large values (which are most likely outliers) severely skew the statistics and do not provide
    # meaningful results. Also loss >> ~20 can more or less just be treated as infinity.
    df[df.filter(regex=r'train_loss_\d{2}') > 100] = np.nan
    df[df.filter(regex=r'val_loss_\d{2}') > 100] = np.nan

    for metric in ['train_loss', 'train_scores', 'val_loss', 'val_scores']:
        df[metric + '_mean'] = df.filter(regex=metric + r'_\d{2}').mean(axis=1)
        df[metric + '_median'] = df.filter(regex=metric + r'_\d{2}').median(axis=1)
        df[metric + '_std'] = df.filter(regex=metric + r'_\d{2}').std(axis=1)

    df = df.sort_values(by=['train_loss_mean', 'val_loss_mean'], ascending=True)
    a = df.loc[df.index[0]].filter(regex=r'train_loss_\d{2}')
    # The filter function casts fields to `object` type for some reason???
    a = a.astype(np.float)

    N = len(df)

    for n, i in enumerate(df.index):
        print('\rPerforming t-tests %d/%d...' % (n + 1, N), end='')
        b = df.loc[i].filter(regex=r'train_loss_\d{2}')
        b = b.astype(np.float)

        if not b.dropna().empty:
            t_statistic, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
            df.loc[i, 't_statistic'] = t_statistic
            df.loc[i, 'p_value'] = p_value

        df.loc[i, 'n_a'] = len(a.dropna())
        df.loc[i, 'n_b'] = len(b.dropna())

    df = df.sort_values(by=['t_statistic', 'p_value', 'n_a', 'n_b'], ascending=[False, True, False, False])
    df.to_csv(output_path)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyses the results based on the pandas dataframe generated from '
                                                 'the `generate_df.py` script, A series of statistical tests '
                                                 'are performed on the data.')
    parser.add_argument('-d', '--data', type=str, default='results_overview.csv',
                        help='Where the file that contains the summarised results is located.')
    parser.add_argument('-o', '--output', type=str, default='results_analysis.csv',
                        help='Where to save the dataframe produced by the analysis.')

    args = parser.parse_args()

    # Get reference to the generated dataframe in case the user is using python interactively.
    df = main(args.data, args.output)
