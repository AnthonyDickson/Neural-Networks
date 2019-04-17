import argparse
import hashlib
import json
import multiprocessing
import os
from datetime import datetime
from time import time

import numpy as np
from sklearn import utils
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold

import mlp
from mlp.activation_functions import Identity, Sigmoid, Softmax
from mlp.layers import DenseLayer, GaussianNoise
from mlp.losses import BinaryCrossEntropy, CategoricalCrossEntropy, RMSE
from mlp.network import MLPRegressor, EarlyStopping


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
    def __init__(self, run_id, clf, params, train_loss, train_scores, val_loss, val_scores):
        if isinstance(params['clf_type'], type):
            params['clf_type'] = params['clf_type'].__name__

        if isinstance(params['activation_func'], type):
            params['activation_func'] = params['activation_func'].__name__

        self.run_id = run_id
        self.clf = clf
        self.params = params
        self.train_loss = train_loss
        self.train_scores = train_scores
        self.val_loss = val_loss
        self.val_scores = val_scores

    def __copy__(self):
        return ResultSet(self.run_id, self.clf, self.params,
                         self.train_loss, self.train_scores, self.val_loss, self.val_scores)

    def to_json(self):
        masked_train_loss = np.ma.masked_invalid(self.train_loss)
        masked_train_scores = np.ma.masked_invalid(self.train_scores)
        masked_val_loss = np.ma.masked_invalid(self.val_loss)
        masked_val_scores = np.ma.masked_invalid(self.val_scores)

        train_loss_min = masked_train_loss.min().tolist()
        train_scores_max = masked_train_scores.max().tolist()
        val_loss_min = masked_val_loss.min().tolist()
        val_scores_max = masked_val_scores.max().tolist()

        return {
            'run_id': self.run_id,
            'params': self.params,
            'train_loss_min': train_loss_min,
            'train_scores_max': train_scores_max,
            'val_loss_min': val_loss_min,
            'val_scores_max': val_scores_max
        }

    def save(self, path, subdir=None):
        if not path.endswith('/'):
            path += '/'

        run_path = path + (self.run_id if not subdir else subdir) + '/'
        os.makedirs(run_path, exist_ok=True)

        with open(run_path + 'statistics.json', 'w') as file:
            json.dump(self.to_json(), file)

        np.save(run_path + 'train_loss', self.train_loss)
        np.save(run_path + 'train_scores', self.train_scores)
        np.save(run_path + 'val_loss', self.val_loss)
        np.save(run_path + 'val_scores', self.val_scores)
        self.clf.save(run_path + 'model.json')
        self.clf.save_weights(run_path + 'weights')


def load_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        grid_search_cfg = json.load(f)

        if 'clf_type' in grid_search_cfg:
            for i, clf_type in enumerate(grid_search_cfg['clf_type']):
                class_ = getattr(mlp.network, clf_type)

                grid_search_cfg['clf_type'][i] = class_

        if 'activation_func' in grid_search_cfg:
            for i, activation_type in enumerate(grid_search_cfg['activation_func']):
                class_ = getattr(mlp.activation_functions, activation_type)

                grid_search_cfg['activation_func'][i] = class_

        print('Loaded configuration file: %s.\n' % cfg_file)

    return grid_search_cfg


def validate_datasets(data_dir, config):
    print('Scanning \'%s\' directory for data sets...' % data_dir)

    try:
        _, datasets, _ = next(os.walk(data_dir))
        datasets = sorted(datasets)
        print('Found the following data sets: ' + ', '.join(datasets))

        for dataset in config['dataset']:
            assert dataset in datasets, "The dataset '%s' was found in the configuration files, but it was " \
                                        "not found in the data directory '%s'." % (dataset, data_dir)
    except StopIteration:
        print('The data directory \'%s\' was not found.' % data_dir)
        return False

    return True


def get_model(shape, output_layer_activation_func, loss_func, param_set):
    input_shape, hidden_layer_size, output_layer_size = shape

    return param_set['clf_type']([
        GaussianNoise(input_shape, n_inputs=input_shape, std=param_set['gaussian_noise']),
        DenseLayer(hidden_layer_size, activation_func=param_set['activation_func']()),
        DenseLayer(output_layer_size, activation_func=output_layer_activation_func)
    ], learning_rate=param_set['learning_rate'], momentum=param_set['momentum'], loss_func=loss_func)


def pad(a, length, fill_value=float('-inf')):
    if len(a) == length:
        return a

    temp = np.zeros(length)
    temp.fill(fill_value)
    temp[:len(a)] = a

    return temp


def evaluation_step(clf, batch_size, shuffle_batches, X_train, y_train, val_set=0, n_epochs=10000):
    es = EarlyStopping(patience=1000, criterion=0.99)

    train_loss, train_score, val_loss, val_score = clf.fit(X_train, y_train, val_set=val_set,
                                                           n_epochs=n_epochs, batch_size=batch_size,
                                                           shuffle_batches=shuffle_batches, early_stopping=es,
                                                           log_verbosity=0)

    train_loss = pad(train_loss, n_epochs)
    train_score = pad(train_score, n_epochs)
    val_loss = pad(val_loss, n_epochs)
    val_score = pad(val_score, n_epochs)
    score = clf.score(*val_set) if val_set != 0 else clf.score(X_train, y_train)

    return score, train_loss, train_score, val_loss, val_score


def run_grid_search(args):
    np.random.seed(args.random_seed)

    data_dir = args.data_dir
    results_dir = args.results_dir
    n_trials = args.n_trials
    n_splits = args.n_splits
    n_jobs = args.n_jobs if args.n_jobs > 0 else len(os.sched_getaffinity(0))

    grid_search_cfg = load_cfg(args.config)

    if not validate_datasets(data_dir, grid_search_cfg):
        exit(1)

    param_grid = ParameterGrid(grid_search_cfg)
    n_param_sets = len(param_grid)
    total_steps = n_param_sets * n_trials

    start = datetime.now()
    print('Grid Search started at: %s' % start)
    print('Grid Search running with %d job(s).' % n_jobs)

    for i, param_set in enumerate(param_grid):
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

        utils.shuffle(X, y, y_multiclass)

        hidden_layer_size = ParamSet(raw_params).hidden_layer_size
        output_layer_size = y.shape[1]
        shape = (X.shape[1], hidden_layer_size, output_layer_size)

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
        clf = None

        if param_set['dataset'] == 'iris':
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_trials // n_splits)

            for train_index, val_index in cv.split(X, y_multiclass):
                X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

                clf = get_model(shape, output_layer_activation_func, loss_func, param_set)
                batches.append((clf, param_set['batch_size'], param_set['shuffle_batches'],
                                X_train, y_train, (X_val, y_val)))
        else:
            for n in range(n_trials):
                clf = get_model(shape, output_layer_activation_func, loss_func, param_set)
                X_train, y_train = utils.shuffle(X, y)
                batches.append((clf, param_set['batch_size'], param_set['shuffle_batches'],
                                X_train, y_train))

        with multiprocessing.Pool(n_jobs) as p:
            p_results = p.starmap(evaluation_step, batches)

        best_param_set_score = -2 ** 32 - 1
        best_param_set_clf = None
        train_loss_history = []
        train_scores = []
        val_loss_history = []
        val_scores = []

        for batch_i, (score, train_loss, train_score, val_loss, val_score) in enumerate(p_results):
            if score > best_param_set_score and score != float('nan') and abs(score) != float('inf'):
                best_param_set_score = score
                best_param_set_clf = batches[batch_i][0]

            train_loss_history.append(train_loss)
            train_scores.append(train_score)
            val_loss_history.append(val_loss)
            val_scores.append(val_score)

        results = ResultSet(run_id, best_param_set_clf if best_param_set_clf else clf, param_set,
                            train_loss_history, train_scores, val_loss_history, val_scores)

        subdir = '/'.join(['%s=%s' % (key, str(value)) for key, value in zip(param_set.keys(), param_set.values())])
        results.save(results_dir, subdir)

        curr_step = (i + 1) * n_trials
        print('\rProgress: %d/%d (%05.2f%%) - Elapsed time: %s'
              % (curr_step, total_steps, 100 * curr_step / total_steps, datetime.now() - start),
              end='')

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search on various MLP configurations and datasets.')
    parser.add_argument('config', type=str, help='The configuration file to use for this experiment. '
                                                 'See `generate_grid_search_cfgs.py`,')
    parser.add_argument('--data-dir', type=str, default='data/', help='Where the data sets are located.')
    parser.add_argument('--results-dir', type=str, default='results/', help='Where to save the results to.')
    parser.add_argument('--n-trials', type=int, default=20, help='How many times to repeat each configuration.')
    parser.add_argument('--n-splits', type=int, default=5, help='How many folds to use for cross validation.')
    parser.add_argument('--n-jobs', type=int, default=1, help='How many processors to use.')
    parser.add_argument('--random-seed', type=int, default=42, help='The seed for the random number generator.')

    args = parser.parse_args()

    run_grid_search(args)
