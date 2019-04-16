import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('results_summary.csv', index_col=0)

    ##############################
    # Add stats on failure rate. #
    ##############################
    # Failure for a regression model is when the RMSE loss gets stuck at 0.5. This does not apply to all models and datasets
    # but applies mainly to the datasets where the output space is in the set {0, 1}^1, i.e. a single binary value.
    df_regression = df[df['clf_type'] == 'MLPRegressor']
    train_loss_cols = df.filter(regex='train_loss_\d{2}').columns.values

    # axis=1 takes the sum along the rows (axis=0 does this along the columns)
    fail_count = (np.abs(df_regression[train_loss_cols] - 0.5) < 0.01).sum(axis=1) + \
                 (df_regression[train_loss_cols].isna().sum(axis=1)) 
    fail_rate = fail_count / len(train_loss_cols)
    df_regression = df_regression.assign(fail_count=fail_count, fail_rate=fail_rate)

    # Failure for a classification model is when the accuracy gets stuck at 0.5. This does not apply to all models and datasets
    # but applies mainly to the datasets where the output space is in the set {0, 1}^1, i.e. a single binary value.                   
    df_classification = df[df['clf_type'] == 'MLPClassifier']
    train_scores_cols = df.filter(regex='train_scores_\d{2}').columns.values

    fail_count = (np.abs(df_classification[train_scores_cols] - 0.5) < 0.01).sum(axis=1) + \
                 (df_classification[train_scores_cols].isna().sum(axis=1))
    fail_rate = fail_count / len(train_scores_cols)
    df_classification = df_classification.assign(fail_count=fail_count, fail_rate=fail_rate)

    df = pd.concat([df_regression, df_classification])

    #####################
    # Add summary stats #
    #####################
    train_loss_cols = df.filter(regex='train_loss_\d{2}').columns.values
    train_scores_cols = df.filter(regex='train_scores_\d{2}').columns.values
    val_loss_cols = df.filter(regex='val_loss_\d{2}').columns.values
    val_scores_cols = df.filter(regex='val_scores_\d{2}').columns.values

    df['mean_train_loss'] = df[train_loss_cols].mean(axis=1)
    df['mean_train_scores'] = df[train_scores_cols].mean(axis=1)
    df['mean_val_loss'] = df[train_loss_cols].mean(axis=1)
    df['mean_val_scores'] = df[train_scores_cols].mean(axis=1)
    
    df['median_train_loss'] = df[train_loss_cols].median(axis=1)
    df['median_train_scores'] = df[train_scores_cols].median(axis=1)
    df['median_val_loss'] = df[train_loss_cols].median(axis=1)
    df['median_val_scores'] = df[train_scores_cols].median(axis=1)
    
    df['std_train_loss'] = df[train_loss_cols].std(axis=1)
    df['std_train_scores'] = df[train_scores_cols].std(axis=1)
    df['std_val_loss'] = df[train_loss_cols].std(axis=1)
    df['std_val_scores'] = df[train_scores_cols].std(axis=1)

    #########################
    # Update dataframe file #
    #########################
    df.to_csv('results_summary.csv')