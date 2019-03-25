import os
import sys

sys.path.append(os.path.abspath('../'))

import argparse
import json

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn import utils

from mlp.layers import DenseLayer
from mlp.network import MLP


def forward_pass_viz(g):
    # Inputs
    g.attr('node', shape='square')
    nodes = [[]]

    for j in range(len(X[sample])):
        nodes[0].append('x_%d' % j)

    g.attr('node', shape='circle')
    g.node('bias', label='bias=1')

    # Layers
    for l in range(len(model.layers)):
        layer = model.layers[l]

        # Skip misc layers.
        if not isinstance(layer, DenseLayer):
            continue

        # Units in layer
        nodes.append([])

        with g.subgraph(name='cluster_layer_%d' % l,
                        body=['label = "%s";' % layer.name]) as sg:
            for j in range(layer.n_units):
                # Add unit + preactvation and activation values.
                nodes[-1].append('%d_%d' % (l, j))
                sg.node(nodes[-1][-1], label='<net<SUB>{0:d},{1:d}</SUB>={2:.4f}<BR/>'
                                             '  o<SUB>{0:d},{1:d}</SUB>={3:.4f}>'.format(
                    l, j, layer.preactivation_value[0, j],
                    layer.activation_value[0, j]))

                # Add edges from previous layer to this unit.
                for i, node in enumerate(nodes[-2]):
                    sg.edge(node, nodes[-1][-1],
                            label='<W<SUB>%d,%d</SUB>=%.4f>' % (l, j, layer.W[i, j]))
                    # label='%.4f' % layer.W[i, j])

                # Add bias edge.
                sg.edge('bias', nodes[-1][-1], label='%0.4f' % layer.b[0, j])

    return g


def backward_pass_viz(g):
    # Inputs
    nodes = [[] for _ in range(len(model.layers))]

    # Network
    g.attr('node', shape='circle')

    l = len(model.layers) - 1
    i = 0

    layer = model.layers[l]

    with g.subgraph(name='cluster_layer_%d' % l,
                    body=['label = "%s";' % layer.name]) as sg:

        for j in range(layer.n_units):
            nodes[l].append('bp%d_%d' % (l, j))

            delta = error_grads[i] * layer.activation_func.derivative(layer.activation_value)

            sg.node(nodes[l][j], label='<&#948;<SUB>{0:d},{1:d}</SUB>={2:.4f}>'.format(l, j, delta[0, j]))

            sg.edge('error', nodes[l][j])
            sg.edge(nodes[l][j], 'bias',
                    label='<&#916;b<SUB>%d,%d</SUB>=%.4f>' % (l, j, layer.prev_db[j]))

        i += 1

    last_layer = l

    for l in range(len(model.layers) - 2, 0, -1):
        # Skip misc layers.
        layer = model.layers[l]

        if not isinstance(layer, DenseLayer):
            continue

        # Units in layer
        nodes.append([])

        with g.subgraph(name='cluster_layer_%d' % l,
                        body=['label = "%s";' % layer.name]) as sg:

            for j in range(layer.n_units):
                nodes[l].append('bp%d_%d' % (l, j))

                delta = error_grads[i] * layer.next_layer.W.T * layer.activation_func.derivative(
                    layer.activation_value)

                sg.node(nodes[l][j], label='<&#948;<SUB>{0:d},{1:d}</SUB>={2:.4f}>'.format(l, j, delta[0, j]))

                # Add edges from previous layer to this unit.
                for i, node in enumerate(nodes[last_layer]):
                    g.edge(node, nodes[l][j],
                           label='<&#916;W<SUB>%d,%d</SUB>=%.4f>' % (
                               last_layer, j, model.layers[last_layer].prev_dW[j, i]))

                # Add bias edge.
                sg.edge(nodes[l][j], 'bias',
                        label='<&#916;b<SUB>%d,%d</SUB>=%.4f>' % (l, j, layer.prev_db[j]))

        last_layer = l
        i += 1

    g.attr('node', shape='square')

    for i in range(len(X[sample])):
        for j in range(model.layers[last_layer].n_units):
            g.edge(nodes[last_layer][j], 'x_%d' % i,
                   label='<&#916;W<SUB>%d,%d</SUB>=%.4f>' % (l, j, model.layers[last_layer].prev_dW[i, j]))

    return g


def inputs_viz(g):
    g.attr('node', shape='square')

    for j in range(len(X[sample])):
        g.node('x_%d' % j, label='<x<SUB>%d</SUB>=%.2f>' % (j, X[sample, j]))

    g.node('bias', label='bias=1')

    for i in range(y.shape[1]):
        g.node('y_%d' % i, label='<y<SUB>%d</SUB>=%.2f>' % (i, y[sample, i]))


def outputs_viz(sg):
    sg.attr('node', shape='square')

    # Error
    sg.node('error', label='e=%s' % error)

    for i in range(y.shape[1]):
        sg.edge('y_%d' % i, 'error')

    # Loss
    sg.node('loss', label='loss=%.4f' % loss)
    sg.edge('error', 'loss')

    # Connect neurons in last layer.
    sg.attr('node', shape='circle')
    l = len(model.layers) - 1

    for j in range(model.layers[l].n_units):
        sg.node('%d_%d' % (l, j), label='<o<SUB>%d,%d</SUB>=%.4f>' % (l, j, model.layers[l].activation_value[0, j]))
        sg.edge('%d_%d' % (l, j), 'error')


def plot(loss_history):
    plt.plot(loss_history, label='train')
    plt.xlim(0)
    plt.xlabel('Epoch')
    plt.ylabel('%s Loss' % model.loss_func.__class__.__name__)
    plt.title('Loss vs. Epochs for model loaded from \'%s\'' % args.model_file)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through the training process of a MLP.')
    parser.add_argument('-m', '--model-file', type=str, default='xor_model.json',
                        help='The json file describing the model to use.')
    parser.add_argument('-d', '--dataset-path', type=str, default='../data/xor', help='The path to a dataset.')
    parser.add_argument('-s', '--shuffle', action='store_true',
                        help='Flag indicating that the dataset should be shuffled.')

    args = parser.parse_args()
    np.set_printoptions(precision=4)

    with open(args.model_file, 'r') as f:
        model_json = json.load(f)

    model = MLP.from_json(model_json)

    X = np.genfromtxt(args.dataset_path + '/in.txt')
    y = np.genfromtxt(args.dataset_path + '/teach.txt')

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    params = np.genfromtxt(args.dataset_path + '/params.txt')

    if args.shuffle:
        X, y = utils.shuffle(X, y)

    skip = False
    epoch = 0
    skip_epochs = 0
    loss_history = []
    N = len(X)

    def prompt():
        global skip, skip_epochs

        if not skip:
            cmd = input('Press enter to continue or `s` to skip to next epoch:')

            if cmd.lower() == 's':
                skip_epochs = 1
                skip = True


    while True:
        if not skip:
            print('*' * 80)
            print('Epoch: %d' % (epoch + 1))
            print('Epoch score: %.4f' % model.score(X, y))

            cmd = input('Enter `q` to quit, '
                        '`skip n` to skip forward n epochs, '
                        '`plot` to plot the loss history so far, '
                        'or press enter continue: ')

            cmd = cmd.lower()
            parts = cmd.split()

            if cmd == 'q':
                break
            elif cmd == 'plot':
                plot(loss_history)
            elif parts and parts[0] == 'skip' and len(parts) == 2:
                try:
                    skip_epochs = int(parts[1])
                except ValueError:
                    skip_epochs = input('\'%s\' is not a number. Enter an integer:' % str(parts[1]))
                    skip_epochs = int(skip_epochs)

                print('Skipping forward %d epochs...' % skip_epochs)
                skip = True

            print()

        loss_history.append(0)

        for sample in range(N):
            y_pred = model._forward(X[sample].reshape(1, -1))
            error = y[sample] - y_pred
            loss = model.loss_func(y[sample], y_pred)
            loss_history[epoch] += loss / len(X)

            g = graphviz.Digraph()
            g.attr('graph', rankdir='DU')

            if not skip:
                with g.subgraph(name='cluster_inputs', body=['label = "Inputs";']) as sg:
                    inputs_viz(sg)

                with g.subgraph(name='cluster_forward', body=['label = "Forward Pass";']) as sg:
                    forward_pass_viz(sg)

                with g.subgraph(name='cluster_outputs', body=['label = "Outputs";']) as sg:
                    outputs_viz(sg)

                g.render('viz/epoch_%02d/sample_%02d/fp.gv' % (epoch + 1, sample + 1))
                g.render('viz/temp.gz', view=True)  # print to temp file so the same window can be used.
                print('Forward pass of sample %d' % sample)
                prompt()

            error_grads = []

            for layer in reversed(model.layers):
                if not error_grads:
                    error_grads.append(layer.backward(model.loss_func.grad))
                else:
                    error_grads.append(layer.backward(error_grads[-1]))

            if not skip:
                g = graphviz.Digraph()
                g.attr('graph', rankdir='DU')

                with g.subgraph(name='cluster_inputs', body=['label = "Inputs";']) as sg:
                    inputs_viz(sg)

                with g.subgraph(name='cluster_outputs', body=['label = "Outputs";']) as sg:
                    outputs_viz(sg)

                with g.subgraph(name='cluster_backward_pass', body=['label = "Backward Pass";']) as sg:
                    backward_pass_viz(sg)

                g.render('viz/epoch_%02d/sample_%02d/bp.gv' % (epoch + 1, sample + 1))
                g.render('viz/temp.gz', view=True)
                print('Backward pass of sample %d' % sample)
                prompt()

        epoch += 1

        if skip:
            skip_epochs -= 1

            if skip_epochs == 0:
                skip = False
