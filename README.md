# About This Project
This repository contains an implemention of artifical neural networks (see this [Blog Post](https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9) and [Wikipeda page](https://en.wikipedia.org/wiki/Artificial_neural_network) for a brief introduction to neural networks) and the backpropagation learning algorithm coded from scratch only relying on NumPy and a few utility functions from scikit-learn. The API that I build is similar to parts of the scikit-learn and Keras APIs which focuses on an easy-to-use object-oriented interface for building machine learning models.

I ran experiments to explore the effects of different hyperparameters and settings through a grid search. I collected and analysed the data from these experiments and summarised my findings in a technical report (see the file [COSC420_Assignment_1_Report.pdf](https://github.com/eight0153/Neural-Networks/blob/master/COSC420_Assignment_1_Report.pdf)). Copied from the abstract of my report:
> Neural networks are a family of complex learning algorithms. One facet of this complexity is
> choosing the correct hyperparameters such as
> learning rate, momentum, and batch size. For
> this assignment I perform a large scale parameter
> search and explore the effects of certain hyperparameters and other various settings. Through
> my experiments I investigate the validity of several rules of thumb regarding these hyperparameters and settings, and the majority of them
> prove to be sound advice in the context of the
> Iris data set.

This project was done as part of the paper COSC420 at the University of Otago in semester one 2019.

## Getting Started
1.  Set up your python environment.

    If you are using conda you can do this by running the following command:
    ```bash
    $ conda env create -f environment.yml
    ```
    This will create a new conda environment called 'cosc420' which should be used for running code in this project.
    You can change the `name` field of `environment.yml` if the environment name conflicts with any existing environments.

    Otherwise, ensure that your python environment has the packages listed in `environment.yml` installed.

2.  For running the back-propagation visualisation demo in `demos/backprop_demo_graphviz.py` you will need a working installation of GraphViz.
    This can be found [here](https://graphviz.gitlab.io/download/).

3.  Try out one of demos in `demos`!
    You need to `cd` into the `demos` directory before running any of the python programs.

    For `demos/backprop_demo.py` and `demos/backprop_demo_graphviz.py` you can use the command line option `-h`
    to show the help text.
