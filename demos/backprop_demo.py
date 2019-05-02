import os
import sys

sys.path.append(os.path.abspath('../'))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.layers import DenseLayer
from mlp.network import MLP


class Demo:
    def __init__(self, model_file, dataset_path, shuffle, log_frequency=100, max_display_rows=4):
        self.model_file = model_file

        with open(model_file, 'r') as f:
            self.model_json = json.load(f)

        self.X = np.genfromtxt(dataset_path + '/in.txt')
        self.y = np.genfromtxt(dataset_path + '/teach.txt')

        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

        if shuffle:
            self.X, self.y = utils.shuffle(self.X, self.y)

        self.input_dims = self.X.shape[1]

        self.log_frequency = log_frequency
        self.max_display_rows = max_display_rows

        self.epoch = 0
        self.skip_epochs = 0
        self.skipping = False
        self.should_quit = False

        self.y_pred = np.array([])
        self.logits = np.array([])
        self.loss_history = np.array([])

        self.model = MLP.from_json(self.model_json)

    def reset(self):
        self.epoch = 0
        self.skip_epochs = 0
        self.skipping = False
        self.should_quit = False
        self.logits = np.array([])
        self.loss_history = []

        self.model = MLP.from_json(self.model_json)

    def prompt(self):
        if not self.skipping:
            input('Press enter to continue.')
            print()

    def display(self, *args):
        if not self.skipping:
            print(*args)

    def get_matrix_display_string(self, matrix):
        if len(matrix) <= self.max_display_rows:
            return str(matrix)

        n_omitted_rows = len(matrix) - self.max_display_rows

        return str(matrix[:self.max_display_rows]) + '\n%d rows omitted...' % n_omitted_rows

    def plot(self):
        plt.plot(self.loss_history, label='train')
        plt.xlim(0)
        plt.xlabel('Epoch')
        plt.ylabel('%s Loss' % self.model.loss_func.__class__.__name__)
        plt.title('Loss vs. Epochs for model loaded from \'%s\'' % self.model_file)
        plt.legend()
        plt.show()

    def test_loop(self):
        while True:
            input_pattern = input('Enter an input pattern (%d numbers) or \'q\' to quit: ' % self.input_dims)
            nums = input_pattern.lower().split()

            if input_pattern == 'q':
                break

            try:
                if len(nums) < self.input_dims:
                    raise ValueError

                nums = list(map(lambda string: float(string), nums))
            except ValueError:
                print('Invalid input: input should be %d numbers separated by spaces.' % self.input_dims)
                continue

            self.forward_pass(np.array([nums]))

            print('Network Prediction:')
            print(self.model.predict(nums))

    def forward_pass(self, X):
        self.display('************')
        self.display('Forward pass')
        self.display('************')
        output = X
        self.display('Input:\n', self.get_matrix_display_string(output), '\n')

        for layer in self.model.layers:
            self.display('Output of layer:', layer)
            output = layer.forward(output)
            self.display(self.get_matrix_display_string(output))
            self.display()

    def backward_pass(self):
        self.display('*************')
        self.display('Backward Pass')
        self.display('*************')

        if not self.skipping:
            self.model.loss_func(self.y, self.model._forward(self.X))

            error_grad = self.model.loss_func.grad

            for layer in reversed(self.model.layers):
                if isinstance(layer, DenseLayer):
                    self.display('Error of %slayer:' % ('output ' if layer.is_output else ''), layer)
                    self.display(self.get_matrix_display_string(error_grad))
                    error_grad = layer.backward(error_grad)

                    self.display('dW:\n', layer.prev_dW)
                    self.display('db:\n', layer.prev_db)
                    self.display()

        else:
            for i in range(len(self.X)):
                self.model.loss_func(self.y[i:i + 1], self.model._forward(self.X[i:i + 1]))
                self.model._backward()

    def epoch_summary(self):
        self.display('*************')
        self.display('Epoch Results')
        self.display('*************')

        score = self.model.score(self.X, self.y)

        loss = self.model.loss_func(self.y, self.model._forward(self.X, is_training=False))
        loss = loss.mean()
        self.loss_history = np.append(self.loss_history, loss)

        self.y_pred = self.model.predict(self.X)

        self.display('Output:\n', self.get_matrix_display_string(self.y_pred))
        self.display('Teacher:\n', self.get_matrix_display_string(self.y))

        if self.skipping and self.epoch % self.log_frequency == 0:
            print('Epoch %d - Loss: %.4f - Score: %.4f' % (self.epoch, loss, score))
        else:
            self.display('Epoch loss: %.4f' % loss)
            self.display('Epoch score: %.4f' % score)
            self.display()

    def command_loop(self):
        exit_loop = False

        while not exit_loop:
            print('*' * 80)
            print('Epoch: %d' % (self.epoch + 1))
            print('*' * 80)

            cmd = input('Enter `q` to quit, \n'
                        '`reset` to reset the network, \n'
                        '`train n` to train for n epochs, \n'
                        '`weights` to print the network weights, \n'
                        '`test` to manually input a pattern, \n'
                        '`plot` to plot the loss history so far, \n'
                        'or press enter continue: ')

            cmd = cmd.strip()
            cmd = cmd.lower()
            parts = cmd.split()

            print()

            if not cmd:
                exit_loop = True
            elif cmd == 'q':
                self.should_quit = True

                exit_loop = True
            elif cmd == 'reset':
                self.reset()
            elif cmd == 'plot':
                self.plot()
            elif cmd == 'weights':
                for layer in self.model.layers:
                    if isinstance(layer, DenseLayer):
                        print('Weights for:', layer)
                        print('W:\n', layer.W)
                        print('b:\n', layer.b)
                        print()
            elif cmd == 'test':
                self.test_loop()
            elif parts and parts[0] == 'train' and len(parts) == 2:
                try:
                    self.skip_epochs = int(parts[1])
                except ValueError:
                    self.skip_epochs = input('\'%s\' is not a number. Enter an integer:' % str(parts[1]))
                    self.skip_epochs = int(self.skip_epochs)

                if self.skip_epochs > 0:
                    print('Training for %d epochs...' % self.skip_epochs)
                    self.skipping = True

                exit_loop = True
            else:
                print('Unrecognised command \'%s\'.' % cmd)

    def main_loop(self):
        while not self.should_quit:
            if not self.skipping:
                self.command_loop()

                if self.should_quit:
                    break

            self.forward_pass(self.X)
            self.prompt()

            self.backward_pass()
            self.prompt()

            self.epoch_summary()
            self.prompt()

            self.display('End of epoch %d.' % (self.epoch + 1))
            self.epoch += 1

            if self.skipping:
                self.skip_epochs -= 1

                if self.skip_epochs == 0:
                    self.skipping = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through the training process of a MLP.')
    parser.add_argument('-m', '--model-file', type=str, default='xor_model.json',
                        help='The json file describing the model to use.')
    parser.add_argument('-d', '--dataset-path', type=str, default='../data/xor', help='The path to a dataset.')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Flag indicating that the dataset should be shuffled.')

    args = parser.parse_args()
    print(args)

    demo = Demo(args.model_file, args.dataset_path, args.shuffle)
    demo.main_loop()
