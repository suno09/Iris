from sklearn.datasets import load_iris
from collections import defaultdict
from itertools import combinations

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import itertools


def plot_3d_iris_features():
    """
    This function returns scatter 3D plots of many figures about combination of
    features.
    This visualization can make a view about choosing the features (all or many)
    :return: best plot is for feature:
        Sepal length, Petal width, Petal length
    """
    iris_data_by_labels = defaultdict(list)
    iris_data = load_iris()

    indices = list(combinations(range(len(iris_data.feature_names)), 3))
    fig = plt.figure()

    # devise the data to label category (3 categories)
    for features_value, label_value in zip(iris_data.data, iris_data.target):
        iris_data_by_labels[label_value].append(features_value)

    # set the color and the design
    axes_infos = [['r', 'o', iris_data_by_labels[0]],
                  ['b', 's', iris_data_by_labels[1]],
                  ['g', '^', iris_data_by_labels[2]]
                  ]

    for indice in range(len(indices)):
        axe = fig.add_subplot(2, 2, indice + 1, projection="3d")
        indice_x, indice_y, indice_z = indices[indice]
        # x_values, y_values, z_values = [], [], []
        for c, m, features_values in axes_infos:
            x, y, z = [[f[i] for f in features_values] for i in indices[indice]]
            # x_values += x
            # y_values += y
            # z_values += z
            axe.scatter(x, y, z, c=c, marker=m)

        # axe.xaxis.set_ticks(np.arange(min(x_values), max(x_values), 0.5))
        # axe.yaxis.set_ticks(np.arange(min(y_values), max(y_values), 0.5))
        # axe.zaxis.set_ticks(np.arange(min(z_values), max(z_values), 0.5))

        # name of each axe
        axe.set_xlabel(iris_data.feature_names[indice_x])
        axe.set_ylabel(iris_data.feature_names[indice_y])
        axe.set_zlabel(iris_data.feature_names[indice_z])

        axe.set_xticks([])
        axe.set_yticks([])
        axe.set_zticks([])

        fig.canvas.set_window_title('Visualization Scatter 3D Iris Features')

    plt.show()


def plot_2d_iris_features():
    """
    This function returns scatter 2D plots of many figures about combination of
    features.
    This visualization can make a view about choosing the features (all or many)
    :return: best plot is for feature:
        Petal width, Petal length
    """
    iris_data_by_labels = defaultdict(list)
    iris_data = load_iris()

    indices = list(combinations(range(len(iris_data.feature_names)), 2))
    fig = plt.figure()

    # devise the data to label category (3 categories)
    for features_value, label_value in zip(iris_data.data, iris_data.target):
        iris_data_by_labels[label_value].append(features_value)

    # set the color and the design
    axes_infos = [['r', iris_data_by_labels[0]],
                  ['b', iris_data_by_labels[1]],
                  ['g', iris_data_by_labels[2]]
                  ]

    for indice in range(len(indices)):
        axe = fig.add_subplot(2, 3, indice + 1)
        indice_x, indice_y = indices[indice]
        # x_values, y_values = [], []
        for c, features_values in axes_infos:
            x, y = [[f[i] for f in features_values] for i in indices[indice]]
            # x_values += x
            # y_values += y
            axe.scatter(x, y, s=10, c=c)

        # axe.xaxis.set_ticks(np.arange(min(x_values), max(x_values), 0.5))
        # axe.yaxis.set_ticks(np.arange(min(y_values), max(y_values), 0.5))

        # name of each axe
        axe.set_xlabel(iris_data.feature_names[indice_x])
        axe.set_ylabel(iris_data.feature_names[indice_y])

        plt.xticks([])
        plt.yticks([])

        fig.canvas.set_window_title('Visualization Scatter 2D Iris Features')

    plt.show()
