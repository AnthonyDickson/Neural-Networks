import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.activation_functions import Sigmoid, Identity
from mlp.network import MLPRegressor, DenseLayer


class ParamSet:
    def __init__(self, params):
        assert len(params) == 6, "Expected six parameters, got %d." % len(params)

        self.n_inputs = int(params[0])
        self.hidden_layer_size = int(params[1])
        self.output_layer_size = int(params[2])
        self.learning_rate = params[3]
        self.momentum = params[4]
        self.error_criterion = params[5]


class Command:
    def __init__(self, name, description='', usage=''):
        self.name = name
        self.description = description
        self.usage = name if usage == '' else usage

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.name + ': ' + self.description

    @property
    def blurb(self):
        return self.name + ': ' + self.description + '\nUsage: ' + self.usage


class HelpCommand(Command):
    def __init__(self):
        super().__init__('help', 'Explain how to use the program.')

    def __call__(self, commands):
        blurbs = [command.blurb for command in commands]
        print('Available commands:\n- ' + '\n- '.join(blurbs))
        print()


class QuitCommand(Command):
    def __init__(self):
        super().__init__('quit', 'Quit the program.')

    def __call__(self, program):
        program.quit()


class LoadDatasetCommand(Command):
    def __init__(self, datasets):
        name = 'load'
        usage = '%s <%s>' % (name, '|'.join(datasets))

        super().__init__(name, description='Load a dataset.', usage=usage)

        self.datasets = datasets

    def __call__(self, dataset, base_dir='data/', shuffle=True):
        assert dataset in self.datasets, 'Unrecognised dataset \'%s\'.' \
                                         ' Dataset must be one of: %s.' % \
                                         (dataset, ', '.join(self.datasets))

        X = np.genfromtxt(base_dir + dataset + '/in.txt')
        y = np.genfromtxt(base_dir + dataset + '/teach.txt')
        params = np.genfromtxt(base_dir + dataset + '/params.txt')

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if shuffle:
            X, y = utils.shuffle(X, y)

        return X, y, params


class TrainNetworkCommand(Command):
    def __init__(self):
        usage = 'train [batch_size]'

        super().__init__('train',
                         description='Train a MLP on the loaded dataset.',
                         usage=usage)

    def __call__(self, network, X, y, n_epochs=100, batch_size=1,
                 early_stopping_patience=0, early_stopping_threshold=1e-2,
                 log_verbosity=100):
        """ X, y, n_epochs=100, batch_size=-1, early_stopping_patience=-1,
            early_stopping_min_improvement=1e-5, early_stopping_threshold=1e-2,
            log_verbosity=1)"""
        loss_history = network.fit(X, y, n_epochs, batch_size,
                                   early_stopping_patience,
                                   early_stopping_threshold=early_stopping_threshold,
                                   log_verbosity=log_verbosity)

        return loss_history


class PlotLossCommand(Command):
    def __init__(self):
        name = 'plot'
        usage = '%s [title]' % name

        super().__init__(name,
                         description='Plot the results of the last training session.',
                         usage=usage)

    def __call__(self, loss_history, title='Loss vs Epoch'):
        # Commands are split into tokens so title may be spread across these
        # tokens. If that is the case we should concatenate these into a string.
        if type(title) is not str:
            title = ' '.join(title)

        plt.plot(loss_history)
        plt.xlim(0)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)

        plt.show()


class SetRandomSeedCommand(Command):
    def __init__(self):
        name = 'set-seed'
        usage = '%s <random seed>' % name

        super().__init__(name, 'Set the random seed. This helps make things more reproducible,', usage)

    def __call__(self, seed=42):
        np.random.seed(seed)


class CLI:
    def __init__(self, commands):
        self.should_quit = False
        self.commands = commands

        self.X = np.array([])
        self.y = np.array([])
        self.params = np.array([])

        self.network = None
        self.loss_history = np.array([])

    def main_loop(self):
        while not self.should_quit:
            try:
                command = input("Enter a command: ")
            except EOFError:
                print()
                break

            command = command.strip()
            command = command.lower()
            parts = command.split()

            if self.is_valid(parts[0]):
                args = parts[1:] if len(parts) > 1 else None
                self.execute(parts[0], args)
            else:
                print("'" + command + "' is not a valid command. Enter 'help' for a list of available commands.")

    def is_valid(self, command_name):
        if command_name == "":
            return False

        for command in self.commands:
            if command.name == command_name:
                break
        else:
            return False

        return True

    def execute(self, command_name, args=None):
        for command in self.commands:
            if command.name == command_name:
                if isinstance(command, HelpCommand):
                    command(self.commands)
                elif isinstance(command, QuitCommand):
                    command(self)
                elif isinstance(command, LoadDatasetCommand):
                    try:
                        self.X, self.y, params_list = command(args[0])

                        self.params = ParamSet(params_list)

                        layers = [
                            DenseLayer(self.params.hidden_layer_size,
                                       n_inputs=self.params.n_inputs,
                                       activation_func=Sigmoid()),

                            DenseLayer(self.params.output_layer_size,
                                       activation_func=Identity())
                        ]

                        self.network = MLPRegressor(layers, learning_rate=self.params.learning_rate,
                                                    momentum=self.params.momentum)
                    except AssertionError as e:
                        print(e)
                elif isinstance(command, TrainNetworkCommand):
                    try:
                        if args:
                            batch_size = int(args[0])
                        else:
                            batch_size = 1

                        self.loss_history = command(self.network,
                                                    self.X, self.y,
                                                    batch_size=batch_size,
                                                    n_epochs=10000,
                                                    early_stopping_threshold=self.params.error_criterion)
                    except AttributeError as e:
                        if self.network is None:
                            print('Model not found. Have you loaded a dataset yet?')
                        else:
                            raise e
                    except ValueError:
                        print('Invalid batch size. Batch size must be an integer.')
                    except KeyboardInterrupt:
                        print()
                elif isinstance(command, PlotLossCommand):
                    if args:
                        command(self.loss_history, args)
                    else:
                        command(self.loss_history)
                elif isinstance(command, SetRandomSeedCommand):
                    if args:
                        try:
                            command(int(args[0]))
                        except ValueError:
                            print('Invalid random seed. Random seed must be an integer.')
                    else:
                        command()

                break

    def quit(self):
        self.should_quit = True


if __name__ == '__main__':
    print('Scanning \'data\' directory for data sets...')

    try:
        _, datasets, _ = next(os.walk('data'))
        print('Found the following data sets: ' + ', '.join(datasets))

        commands = [
            HelpCommand(),
            QuitCommand(),
            LoadDatasetCommand(datasets),
            TrainNetworkCommand(),
            PlotLossCommand(),
            SetRandomSeedCommand()
        ]

        cli = CLI(commands)
        cli.main_loop()
    except StopIteration:
        print('\'data\' directory not found.')
        exit(1)
