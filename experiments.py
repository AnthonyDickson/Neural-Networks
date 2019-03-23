import argparse
import hashlib
import json
import multiprocessing
import os
from datetime import datetime
from time import time

import numpy as np
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold

from mlp.activation_functions import Identity, Sigmoid, Softmax
from mlp.layers import DenseLayer
from mlp.losses import BinaryCrossEntropy, CategoricalCrossEntropy, RMSE
from mlp.network import MLPRegressor, MLPClassifier, EarlyStopping


class ParamSet:
    def __init__(self, params):
        assert len(params) == 6, "Expected six parameters, got %d." % len(params)

        self.n_inputs = int(params[0])
        self.hidden_layer_size = int(params[1])
        self.output_layer_size = int(params[2])
        self.learning_rate = params[3]
        self.momentum = params[4]
        self.error_criterion = params[5]


class ResultSet:
    def __init__(self, run_id, scores, scores_mean, scores_std, params, loss_histories, clf):
        if isinstance(params['clf_type'], type):
            params['clf_type'] = params['clf_type'].__name__

        self.run_id = run_id
        self.scores = scores
        self.scores_mean = scores_mean
        self.scores_std = scores_std
        self.params = params
        self.loss_histories = loss_histories
        self.clf = clf

        if isinstance(self.scores_mean, np.ma.core.MaskedConstant):
            self.scores_mean = float('-inf')

        if isinstance(self.scores_std, np.ma.core.MaskedConstant):
            self.scores_std = float('-inf')

    def __copy__(self):
        return ResultSet(self.run_id, self.scores, self.scores_mean, self.scores_std, self.params, self.loss_histories,
                         self.clf)

    def json(self):
        # TODO: Fix bug with his line. 'ufunc `isinfinite` not supported for the input types.'
        mean_loss_history = np.ma.masked_invalid(self.loss_histories).mean(axis=1).tolist()

        if isinstance(mean_loss_history, np.ma.core.MaskedConstant):
            mean_loss_history = np.zeros_like(self.loss_histories)
            mean_loss_history.fill(float('-nan'))  # would be NaN anyway so doesn't matter what this is set to

        return {
            'run_id': self.run_id,
            'scores': self.scores,
            'scores_mean': self.scores_mean,
            'scores_std': self.scores_std,
            'params': self.params,
            'mean_loss_history': mean_loss_history
        }

    def save(self, path, subdir=None):
        if not path.endswith('/'):
            path += '/'

        run_path = path + (self.run_id if not subdir else subdir) + '/'
        os.makedirs(run_path, exist_ok=True)

        with open(run_path + 'statistics.json', 'w') as file:
            json.dump(self.json(), file)

        np.save(run_path + 'loss_history', self.loss_histories)
        self.clf.save(run_path + 'model.json')
        self.clf.save_weights(run_path + 'weights')


def pad(a, length, fill_value=float('-inf')):
    if len(a) == length:
        return a

    temp = np.zeros(length)
    temp.fill(fill_value)
    temp[:len(a)] = a

    return temp


def evaluation_step(clf, batch_size, shuffle_batches, X_train, X_val, y_train, y_val, n_epochs=10000):
    es = EarlyStopping(patience=100)

    train_loss, train_score, val_loss, val_score = clf.fit(X_train, y_train, val_set=(X_val, y_val),
                                                           n_epochs=n_epochs, batch_size=batch_size,
                                                           shuffle_batches=shuffle_batches, early_stopping=es,
                                                           log_verbosity=100)

    pad(train_loss, n_epochs)
    pad(train_score, n_epochs)
    pad(val_loss, n_epochs)
    pad(val_score, n_epochs)

    return clf.score(X_val, y_val), train_loss, train_score, val_loss, val_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-dir', type=str, default='data/', help='Where the data sets are located.')
    parser.add_argument('--results-dir', type=str, default='results/', help='Where to save the results to.')
    parser.add_argument('--n-trials', type=int, default=20, help='How many times to repeat each configuration.')
    parser.add_argument('--n-jobs', type=int, default=1, help='How many processors to use.')

    args = parser.parse_args()

    np.random.seed(42)

    data_dir = args.data_dir
    results_dir = args.results_dir
    n_trials = args.n_trials
    n_jobs = args.n_jobs if args.n_jobs > 0 else len(os.sched_getaffinity(0))

    datasets = []

    print('Scanning \'%s\' directory for data sets...' % data_dir)

    try:
        _, datasets, _ = next(os.walk('data'))
        datasets = sorted(datasets)
        print('Found the following data sets: ' + ', '.join(datasets))
    except StopIteration:
        print('\'data\' directory not found.')
        exit(1)

    param_grid = ParameterGrid(dict(
        batch_size=[1, 2, 4, -1],
        clf_type=[MLPRegressor, MLPClassifier],
        dataset=datasets,
        learning_rate=[1e0, 1e-1, 1e-2, 1e-3],
        momentum=[0.9, 0.1, 0],
        shuffle_batches=[False, True]
    ))

    best_score = -2 ** 32 - 1
    best_results = None
    n_param_sets = len(param_grid)
    total_steps = n_param_sets * n_trials

    start = datetime.now()
    print('Grid Search Started at: %s' % start)
    print('Grid Search running with %d job(s).' % n_jobs)

    for i, param_set in enumerate(param_grid):
        scores = []
        loss_histories = []
        clf = None
        best_param_set_clf = None

        md5 = hashlib.md5(str(time()).encode('utf-8'))
        run_id = md5.hexdigest()

        dataset = param_set['dataset']
        X = np.genfromtxt(data_dir + dataset + '/in.txt')
        y = np.genfromtxt(data_dir + dataset + '/teach.txt')
        raw_params = np.genfromtxt(data_dir + dataset + '/params.txt')

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            y_multiclass = y
        else:
            y_multiclass = y.argmax(axis=1)
            y_multiclass = y_multiclass.reshape(-1, 1)

        hidden_layer_size = ParamSet(raw_params).hidden_layer_size
        output_layer_size = y.shape[1]

        if param_set['clf_type'] == MLPRegressor:
            output_layer_activation_func = Identity()
            loss_func = RMSE()
        else:
            if output_layer_size == 1:
                output_layer_activation_func = Sigmoid()
                loss_func = BinaryCrossEntropy()
            else:
                output_layer_activation_func = Softmax()
                loss_func = CategoricalCrossEntropy()

        batches = []

        if param_set['dataset'] == 'iris':
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=n_trials)

            for train_index, test_index in cv.split(X, y_multiclass):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

                clf = param_set['clf_type']([
                    DenseLayer(hidden_layer_size, n_inputs=X.shape[1], activation_func=Sigmoid()),
                    DenseLayer(output_layer_size, activation_func=output_layer_activation_func)
                ], learning_rate=param_set['learning_rate'], momentum=param_set['momentum'])

                batches.append((clf, param_set['batch_size'], param_set['shuffle_batches'],
                                X_train, X_test, y_train, y_test))
        else:
            for n in range(n_trials):
                clf = param_set['clf_type']([
                    DenseLayer(hidden_layer_size, n_inputs=X.shape[1], activation_func=Sigmoid()),
                    DenseLayer(output_layer_size, activation_func=output_layer_activation_func)
                ], learning_rate=param_set['learning_rate'], momentum=param_set['momentum'])

                batches.append((clf, param_set['batch_size'], param_set['shuffle_batches'],
                                X, X, y, y))

        with multiprocessing.Pool(n_jobs) as p:
            p_results = p.starmap(evaluation_step, batches)

        best_param_set_score = -2 ** 32 - 1

        # TODO: Record training and testing sets of score and loss history.
        for batch_i, (score, _, _, test_loss_history, _) in enumerate(p_results):
            if score > best_param_set_score and score != float('nan') and score != float('inf'):
                best_param_set_score = score
                best_param_set_clf = batches[batch_i][0]

            scores.append(score)
            loss_histories.append(test_loss_history)

        masked_scores = np.ma.masked_invalid(scores)
        scores_mean = masked_scores.mean()
        scores_std = masked_scores.std()

        results = ResultSet(run_id, scores, scores_mean, scores_std, param_set, loss_histories,
                            best_param_set_clf if best_param_set_clf else clf)
        results.save(results_dir)

        if best_param_set_score > best_score and best_param_set_score > -2:
            best_score = best_param_set_score
            best_results = results.__copy__()

        curr_step = (i + 1) * n_trials
        print('\rProgress: %d/%d (%05.2f%%) - Elapsed time: %s'
              % (curr_step, total_steps, 100 * curr_step / total_steps, datetime.now() - start),
              end='')

    best_results.save(results_dir, subdir='best')
    print('\nDone.')
