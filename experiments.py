import hashlib
import json
import os
from datetime import datetime
from time import time

import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from mlp.activation_functions import Identity, Sigmoid, Softmax
from mlp.layers import DenseLayer
from mlp.losses import BinaryCrossEntropy, CategoricalCrossEntropy, RMSE
from mlp.network import MLPRegressor, MLPClassifier


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

    def __copy__(self):
        return ResultSet(self.run_id, self.scores, self.scores_mean, self.scores_std, self.params, self.loss_histories,
                         self.clf)

    def json(self):
        return {
            'run_id': self.run_id,
            'scores': self.scores,
            'scores_mean': self.scores_mean,
            'scores_std': self.scores_std,
            'params': self.params,
            'mean_loss_history': np.mean(self.loss_histories, axis=1).tolist()
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


# TODO: Make params such as n_trails configurable via command line.
if __name__ == '__main__':
    np.random.seed(42)

    results_dir = 'results/'
    base_dir = 'data/'
    datasets = []

    print('Scanning \'%s\' directory for data sets...' % base_dir)

    try:
        _, datasets, _ = next(os.walk('data'))
        datasets = sorted(datasets)
        print('Found the following data sets: ' + ', '.join(datasets))
    except StopIteration:
        print('\'data\' directory not found.')
        exit(1)

    param_dict = dict(
        dataset=datasets,
        clf_type=[MLPRegressor, MLPClassifier],
        learning_rate=[1e0, 1e-1, 1e-2, 1e-3],
        momentum=[0.99, 0.9, 0.1, 0.01, 0],
        batch_size=[1, 16, 32, -1]
    )

    param_grid = ParameterGrid(param_dict)

    best_score = -2  # -2 since the lowest possible score is -1 (from Pearson's R coefficient).
    best_results = None

    n_trials = 20
    n_param_sets = len(param_grid)
    total_steps = n_param_sets * n_trials

    start = datetime.now()
    print('Grid Search Started at: %s' % start)

    for i, param_set in enumerate(param_grid):
        scores = []
        loss_histories = []
        clf = None
        best_param_set_clf = None

        md5 = hashlib.md5(str(time()).encode('utf-8'))
        run_id = md5.hexdigest()

        dataset = param_set['dataset']
        X = np.genfromtxt(base_dir + dataset + '/in.txt')
        y = np.genfromtxt(base_dir + dataset + '/teach.txt')
        raw_params = np.genfromtxt(base_dir + dataset + '/params.txt')

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

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

        if param_set['dataset'] == 'iris':
            cv = StratifiedKFold(n_splits=2)
        else:
            cv = None

        for n in range(n_trials):
            if cv:
                train_index, test_index = cv.split(X, y)
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            clf = param_set['clf_type']([
                DenseLayer(hidden_layer_size, n_inputs=X.shape[1], activation_func=Sigmoid()),
                DenseLayer(output_layer_size, activation_func=output_layer_activation_func)
            ], learning_rate=param_set['learning_rate'], momentum=param_set['momentum'])

            loss_history = clf.fit(X_train, y_train,
                                   n_epochs=10000, batch_size=param_set['batch_size'],
                                   log_verbosity=0, early_stopping_threshold=-1)
            scores.append(clf.score(X_test, y_test))
            loss_histories.append(loss_history)

            if scores[n] >= max(scores) and scores[n] != float('nan') and scores[n] != float('inf'):
                best_param_set_clf = clf

            curr_step = i * n_trials + (n + 1)
            print('\rProgress: %d/%d (%05.2f%%) - Elapsed time: %s'
                  % (curr_step, total_steps, 100 * curr_step / total_steps, datetime.now() - start),
                  end='')

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)

        results = ResultSet(run_id, scores, scores_mean, scores_std, param_set, loss_histories,
                            best_param_set_clf if best_param_set_clf else clf)
        results.save(results_dir)

        max_score = max(scores)

        if max_score > best_score and scores_mean != float('nan') and scores_mean != float('inf'):
            best_score = max_score
            best_results = results.__copy__()

    best_results.save(results_dir, subdir='best')
    print('\nDone.')
