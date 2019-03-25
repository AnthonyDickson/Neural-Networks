import os
import sys

sys.path.append(os.path.abspath('../'))

import argparse
import json

import numpy as np
from sklearn import utils

from mlp.layers import DenseLayer
from mlp.network import MLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through the training process of a MLP.')
    parser.add_argument('model', metavar='model_file', type=str, help='The json file describing the model to use.')
    parser.add_argument('-d', '--dataset-path', type=str, default='../data/xor', help='The path to a dataset.')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Flag indicating that the dataset should be shuffled.')

    args = parser.parse_args()
    print(args)

    with open(args.model, 'r') as f:
        model_json = json.load(f)

    model = MLP.from_json(model_json)

    X = np.genfromtxt(args.dataset_path + '/in.txt')
    y = np.genfromtxt(args.dataset_path + '/teach.txt')
    y = y.reshape(-1, 1)
    params = np.genfromtxt(args.dataset_path + '/params.txt')

    if args.shuffle:
        X, y = utils.shuffle(X, y)

    epoch = 0
    skip_epochs = 0
    skipping = False


    def prompt():
        if not skipping:
            input('Press enter to continue.')
            print()


    def display(*args):
        if not skipping:
            print(*args)


    while True:
        if not skipping:
            display('*' * 80)
            display('Epoch: %d' % (epoch + 1))

            cmd = input('Enter `q` to quit, '
                        '`skip n` to skip forward n epochs, '
                        '`weights` to print the network weights, '
                        'or press enter continue.')

            cmd = cmd.lower()
            parts = cmd.split()

            if cmd == 'q':
                break
            elif cmd == 'weights':
                for layer in model.layers:
                    if isinstance(layer, DenseLayer):
                        print('Weights for:', layer)
                        print('W:\n', layer.W)
                        print('b:\n', layer.b)
                        print()

                prompt()
            elif parts and parts[0] == 'skip' and len(parts) == 2:
                try:
                    skip_epochs = int(parts[1])
                except ValueError:
                    skip_epochs = input('\'%s\' is not a number. Enter an integer:' % str(parts[1]))
                    skip_epochs = int(skip_epochs)

                print('Skipping forward %d epochs...' % skip_epochs)
                skipping = True

            print()

        display('Forward pass.')
        input_ = X
        display('Input:\n', X, '\n')

        for layer in model.layers:
            display('Output of layer:', layer)
            input_ = layer.forward(input_)
            display(input_)
            display()

        prompt()

        display('Epoch Results.')
        y_pred = input_
        display('Output:\n', y_pred)
        display('Teacher:\n', y)

        loss = model.loss_func(y, y_pred)
        display('Epoch loss (mean): %.4f' % loss.mean())
        error_grad = model.loss_func.grad
        display('Epoch score: %.4f' % model.score(X, y))

        prompt()

        display('Backward Pass.')

        for layer in reversed(model.layers):
            if isinstance(layer, DenseLayer):
                display('Error of %slayer:' % ('output ' if layer.is_output else ''), layer)
                display(error_grad)
                error_grad = layer.backward(error_grad)

                display('dW:\n', layer.prev_dW)
                display('db:\n', layer.prev_db)
                display()

        prompt()

        display('End of epoch %d.' % (epoch + 1))
        display('*' * 80)
        epoch += 1

        if skipping:
            skip_epochs -= 1

            if skip_epochs == 0:
                skipping = False
