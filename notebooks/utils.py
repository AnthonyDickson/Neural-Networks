import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# df = pd.read_csv('../scripts/results_summary_bad_init.csv', index_col=0)


variables = ['activation_func', 'batch_size', 'clf_type', 'dataset', 
             'gaussian_noise', 'learning_rate', 'momentum', 'shuffle_batches']

datasets = ['535', 'encoder', 'iris', 'parity-3-bit', 'parity-4-bit', 'xor']
metrics = ['train_loss', 'train_scores', 'val_loss', 'val_scores']
summary_stats = [stat + '_' + metric for metric, stat in itertools.product(metrics + ['n_epochs'], ['median', 'mean', 'std'])]

# Some useful masks
# Activation Function
def relu_mask(df):
    return df['activation_func'] == 'LeakyReLU'

def sigmoid_mask(df):
    return df['activation_func'] == 'Sigmoid'

# Batch Size
def sgd_mask(df):
    return df['batch_size'] == 1

def minibatch_sgd_mask(df):
    return df['batch_size'] > 1

def batch_sgd_mask(df):
    return df['batch_ size'] == -1  # -1 was the special value used to indicate batch sgd in experiment code.

# Classifier Type
def regression_mask(df):
    return df['clf_type'] == 'MLPRegressor'

def classification_mask(df):
    return df['clf_type'] == 'MLPClassifier'

# Dataset
def _535_mask(df):
    return df['dataset'] == '535'

def encoder_mask(df):
    return df['dataset'] == 'encoder'

def iris_mask(df):
    return df['dataset'] == 'iris'

def parity_3_bit_mask(df):
    return df['dataset'] == 'parity-3-bit'

def parity_4_bit_mask(df):
    return df['dataset'] == 'parity-4-bit'

def xor_mask(df):
    return df['dataset'] == 'xor'

# Shuffling Batches
def get_shuffle_batches(df):
    return df['shuffle_batches'] == True

# Gather groups of column names
def get_train_loss_cols(df):
    return df.filter(regex='train_loss_\d{2}').columns.values

def get_train_scores_cols(df):
    return df.filter(regex='train_scores_\d{2}').columns.values

def get_val_loss_cols(df):
    return df.filter(regex='val_loss_\d{2}').columns.values

def get_val_scores_cols(df):
    return df.filter(regex='val_scores_\d{2}').columns.values

def get_n_epochs_cols(df):
    return df.filter(regex='n_epochs_\d{2}').columns.values

def print_stats(dataframe, name='a'):
    a = dataframe
    print('Summary for %s:' % name)
    print('n: %d' % (len(a.dropna()) * a.shape[1]))
    # Do aggregate functions (e.g. mean()) twice to get column-wise and then datafram-wise measures.
    print('min: %.4e - max: %.4e' % (a.min().min(), a.max().max()))
    print('μ ± 2σ: %.4e ± %.4e' % (a.mean().mean(), 2 * a.std().std()))
    print('Quartiles - 1st: %.4e - 2nd: %.4e - 3rd: %.4e' % (a.quantile(0.25).quantile(0.25),
                                                            a.quantile(0.5).quantile(0.5),
                                                            a.quantile(0.75).quantile(0.75)))
     
        
def format_p_value(p_value):
    return str(p_value % 1)[1:5] if p_value > 0.001 else '<.001'
        
def stat_test(df, base_mask, variable_mask, test_name, alpha=0.001):    
    """Perform a hypothesis test on two subgroups, `a` and `NOT a`.
    
    :param df: The data source dataframe.
    :param base_mask: The base subset of datapoints (e.g. only regression models, only ReLU activation function).
        This should be a boolean mask of `df`.
        
    :param variable_mask: The mask for configuration `a` that you want to compare against `NOT a` 
        (e.g. regression models vs. classification models, ReLU vs Sigmoid).
        This should be a boolean mask of `df`.
    :param alpha: The significance threshold.
    """
    print('*' * 80)
    print(test_name)
    print('*' * 80)
    
    for metric in metrics + ['fail_rate']:
        print('#'  * len(metric))
        print(metric)
        print('#'  * len(metric))
    
        metric_pattern = metric if metric == 'fail_rate' else metric + '_\d{2}'
        a = df[base_mask & variable_mask].filter(regex=metric_pattern)
        b = df[base_mask & ~variable_mask].filter(regex=metric_pattern)
    
        print_stats(a, name='a')
        print()

        print_stats(b, name='b')
        print()
        
        print('Welch t-test:')
        t, p = stats.ttest_ind(a.values.ravel(), b.values.ravel(), equal_var=False, nan_policy='omit')
        print('t: %.4f - p: %s' % (t, format_p_value(p)))


def plot(df, thresholds=(10, -1, 10, -1)):
    """Plot the train loss, train scores, validation loss, and validation scores of a masked dataframe.
    
    :param df: A dataframe that is masked for at least dataset.
    :param thresholds: A 4-tuple of threshold that the [train loss/train scores/validation loss/validation scores]
        must be [lower/higher/lower/higher] than to be plotted, respectively.
        
    :return: The figure and axes of the plot.
    """
    assert len(df['dataset'].unique()) == 1, 'You can only plot one dataset at a time!'
    
    t = thresholds
    train_loss_cols = get_train_loss_cols(df)
    train_scores_cols = get_train_scores_cols(df)
    val_loss_cols = get_val_loss_cols(df)
    val_scores_cols = get_val_scores_cols(df)
    
    # Data Wrangling    
    stacked_train_loss = pd.Series(df[df[train_loss_cols] < t[0]][train_loss_cols].values.ravel())    
    
    stacked_train_scores = pd.Series(df[df[train_scores_cols] > t[1]][train_scores_cols].values.ravel()) 
    
    stacked_val_loss = pd.Series(df[df[val_loss_cols] < t[2]][val_loss_cols].values.ravel()) 
    
    stacked_val_scores = pd.Series(df[df[val_scores_cols] > t[3]][val_scores_cols].values.ravel())
    
    # Plotting
    if len(stacked_val_loss.dropna()) > 0 or len(stacked_val_scores.dropna()) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=True)
    
    stacked_train_loss.hist(bins=100, ax=axes[0])
    axes[0].set_title('Train Loss')
    
    stacked_train_scores.hist(bins=100, ax=axes[1])
    axes[1].set_title('Train Scores')
    
    if len(stacked_val_loss.dropna()) > 0:
        stacked_val_loss.hist(bins=100, ax=axes[2])
        axes[2].set_title('Validation Loss')
    
    if len(stacked_val_scores.dropna()) > 0:
        stacked_val_scores.hist(bins=100, ax=axes[3])
        axes[3].set_title('Validation Scores')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Plot for the Dataset %s' % df.iloc[0]['dataset'])
    
    return fig, axes
    
def plot_hue(masked_df, hue, bins=50, stacked=False, thresholds=(10, -1, 10, -1)):
    """Plot the train loss, train scores, validation loss, and validation scores of a masked dataframe.
    
    :param masked_df: A dataframe that is masked for at least dataset.
    :param hue: The column that will be used to separate the data into groups. For example, 'clf_type' would
        separate the data into two sets for 'MLPRegressor' and 'MLPClassifier'.
    :param thresholds: A 4-tuple of threshold that the [train loss/train scores/validation loss/validation scores]
        must be [lower/higher/lower/higher] than to be plotted, respectively.
        
    :return: The figure and axes of the plot.
    """
    datasets = masked_df['dataset'].unique()
    dataset = datasets[0] if len(datasets) == 1 else None
    
    assert hue == 'dataset' or len(datasets) == 1, 'You can only plot one dataset at a time!'
    
    t = thresholds
    train_loss_cols = get_train_loss_cols(masked_df)
    train_scores_cols = get_train_scores_cols(masked_df)
    val_loss_cols = get_val_loss_cols(masked_df)
    val_scores_cols = get_val_scores_cols(masked_df)
    
    if dataset == 'iris':
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), squeeze=True)
        
    groups = [group for group in masked_df[hue].unique()]
    train_loss_series = []
    train_score_series = []
    validation_loss_series = []
    validation_score_series = []
    
    for group in groups:
        df = masked_df[masked_df[hue] == group]
        
        # Data Wrangling    
        stacked_train_loss = pd.Series(df[df[train_loss_cols] < t[0]][train_loss_cols].values.ravel())    

        stacked_train_scores = pd.Series(df[df[train_scores_cols] > t[1]][train_scores_cols].values.ravel()) 

        stacked_val_loss = pd.Series(df[df[val_loss_cols] < t[2]][val_loss_cols].values.ravel()) 

        stacked_val_scores = pd.Series(df[df[val_scores_cols] > t[3]][val_scores_cols].values.ravel())
        
        train_loss_series.append(stacked_train_loss.dropna())
        train_score_series.append(stacked_train_scores.dropna())
        validation_loss_series.append(stacked_val_loss.dropna())
        validation_score_series.append(stacked_val_scores.dropna())

        
    axes[0].hist(train_loss_series, bins=bins, stacked=stacked, label=groups)
    axes[0].set_title('Train Loss')

    axes[1].hist(train_score_series, bins=bins, stacked=stacked, label=groups)
    axes[1].set_title('Train Scores')

    if dataset == 'iris':
        axes[2].hist(validation_loss_series, bins=bins, stacked=stacked, label=groups)
        axes[2].set_title('Validation Loss')

        axes[3].hist(validation_score_series, bins=bins, stacked=stacked, label=groups)
        axes[3].set_title('Validation Scores')

    axes[-1].legend(title=str(hue).capitalize(), 
                    loc='center left', bbox_to_anchor=(1, 0.5), 
                    fancybox=True, shadow=True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    title = 'Plot of Metrics by %s' % str(hue).capitalize()
    
    if dataset:
        title = title + ' for Dataset %s' % dataset
    
    fig.suptitle(title)
    
    return fig, axes
    
def make_boxplot(df, base_mask, variable_mask, a_config, b_config, metric, title):    
    """Make a box plot comparing two configuration, `a` and `NOT a` (i.e. the opposite of a) 
    in terms of one of the four metrics (train_loss, train_scores, val_loss, val_scores).
    
    :param df: The data source dataframe.
    :param base_mask: The base subset of datapoints (e.g. only regression models, only ReLU activation function).
        This should be a boolean mask of `df`.
        
    :param variable_mask: The mask for configuration `a` that you want to compare against `NOT a` 
        (e.g. regression models vs. classification models, ReLU vs Sigmoid).
        This should be a boolean mask of `df`.
        
    :param a_config: A string representation of the configuration for the variable from variable_mask 
        (e.g. 'activation_func=ReLU').
        
    :param b_config: A string representation of the opposite configuration of the variable from variable_mask 
        (e.g. 'activation_func=Sigmoid').
        
    :param metric: The metric by which the configurations of the variable from variable_mask should be compared.
    :param title: The title of the plot.
    
    :return: The figure and axes of the plot.
    """
    a = df[base_mask & variable_mask].filter(regex='%s_\d{2}' % metric).values
    a = a.ravel()
    a = a[~np.isnan(a)]

    b = df[base_mask & ~variable_mask].filter(regex='%s_\d{2}' % metric).values
    b = b.ravel()
    b = b[~np.isnan(b)]

    ticks = ['%s \n(n=%d)' % (a_config, len(a)), '%s \n(n=%d)' % (b_config, len(b))]
    
    fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(12, 4))
    
    axes[0].boxplot([a, b], showfliers=True)
    axes[0].set_xticklabels(ticks)
    axes[0].set_title(title)

    axes[1].boxplot([a, b], showfliers=False)
    axes[1].set_xticklabels(ticks)
    axes[1].set_title('%s (Outliers Hidden)' % title)
    
    fig.tight_layout()
    
    return fig, axes

def make_n_way_boxplot(df, base_mask, variable_masks, configs, metric, title):    
    """Make box plots comparing multiple values of a variable (e.g. different learning rates)
    in terms of one of the four metrics (train_loss, train_scores, val_loss, val_scores).
    
    :param df: The data source dataframe.
    :param base_mask: The base subset of datapoints (e.g. only regression models, only ReLU activation function).
        This should be a boolean mask of `df`.
        
    :param variable_masks: The masks for the different configurations of the independent variable of interest
        (e.g. learning_rate=0.1, learning_rate=0.01, learning_rate=0.001).
        This should be a boolean mask of `df`.
        
    :param configs: A string representation of the configurations for the variable from variable_masks 
        (e.g. ['learning_rate=0.1', 'learning_rate=0.01', 'learning_rate=0.001']).
        
    :param metric: The metric by which the configurations of the variable from variable_mask should be compared.
    :param title: The title of the plot.
    
    :return: The figure and axes of the plot.
    """
    groups = []

    for subgroup_mask in variable_masks:
        subgroup = df[base_mask & subgroup_mask].filter(regex='%s_\d{2}' % metric).values
        subgroup = subgroup.ravel()
        subgroup = subgroup[~np.isnan(subgroup)]

        groups.append(subgroup)

    ticks = ['%s \n(n=%d)' % (config, len(subgroup)) for config, subgroup in zip(configs, groups)]

    fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(12, 4))

    axes[0].boxplot(groups, showfliers=True)
    axes[0].set_xticklabels(ticks)
    axes[0].set_title(title)

    axes[1].boxplot(groups, showfliers=False)
    axes[1].set_xticklabels(ticks)
    axes[1].set_title('%s (Outliers Hidden)' % title)

    fig.tight_layout()
    
    return fig, axes