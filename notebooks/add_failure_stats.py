import os

import numpy as np
import pandas as pd

df = pd.read_csv('../scripts/results_summary.csv', index_col=0)

# Failure for a regression model is when the RMSE loss gets stuck at 0.5. This does not apply to all models and datasets
# but applies mainly to the datasets where the output space is in the set {0, 1}^1, i.e. a single binary value.
df_regression = df[df['clf_type'] == 'MLPRegressor']
train_loss_cols = df.filter(regex='train_loss_\d{2}').columns.values

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
df.to_csv('../scripts/results_summary.csv')