# COSC420
COSC420 Neural Networks Assignment

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
