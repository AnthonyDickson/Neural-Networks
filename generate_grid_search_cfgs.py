import argparse
import json
import os

from mlp.activation_functions import Sigmoid, LeakyReLU
from mlp.network import MLPRegressor, MLPClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search on various MLP configurations and datasets.')
    parser.add_argument('--data-dir', type=str, default='data/', help='Where the data sets are located.')

    args = parser.parse_args()

    print('Scanning \'%s\' directory for data sets...' % args.data_dir)

    datasets = []

    try:
        _, datasets, _ = next(os.walk('data'))
        datasets = sorted(datasets)
        print('Found the following data sets: ' + ', '.join(datasets))
    except StopIteration:
        print('\'data\' directory not found.')
        exit(1)

    param_grid = dict(
        activation_func=[LeakyReLU.__name__, Sigmoid.__name__],
        batch_size=[1, 2, -1],
        clf_type=[MLPClassifier.__name__, MLPRegressor.__name__],
        dataset=datasets,
        gaussian_noise=[0, 0.01, 0.1],
        learning_rate=[1e-3, 1e-2, 1e-1],
        momentum=[0, 0.5, 0.9],
        shuffle_batches=[False, True]
    )

    main_experiment_filename = 'main_experiment.json'
    iris_experiment_filename = 'additional_iris_experiment.json'
    created_config_fmt = 'Created configuration `%s`.'

    with open(main_experiment_filename, 'w') as f:
        json.dump(param_grid, f)
        print(created_config_fmt % main_experiment_filename)

    if 'iris' not in datasets:
        os.remove(iris_experiment_filename)
    else:
        param_grid['batch_size'] = batch_size = [8, 16, 32]
        param_grid['dataset'] = ['iris']
        param_grid['gaussian_noise'] = [0, 0.01, 0.1]

        with open(iris_experiment_filename, 'w') as f:
            json.dump(param_grid, f)
            print(created_config_fmt % iris_experiment_filename)
