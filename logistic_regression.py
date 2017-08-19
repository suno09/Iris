from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from itertools import combinations

import matplotlib.pyplot as plt

import numpy as np

iris_dataset = load_iris()
logistic_regression = LogisticRegression(C=1e5)


def iris_logistic_regression_two_features():
    inputs = iris_dataset.data
    targets = iris_dataset.target
    indices = list(combinations(range(len(iris_dataset.feature_names)), 2))

    fig = plt.figure()

    for indice in range(len(indices)):
        f1, f2 = indices[indice]

        logistic_regression.fit(inputs[:, [f1, f2]], targets)

        x_min, x_max = inputs[:, f1].min() - .5, inputs[:, f1].max() + .5
        y_min, y_max = inputs[:, f2].min() - .5, inputs[:, f2].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                             np.arange(y_min, y_max, .02))
        Z = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        axe = fig.add_subplot(2, 3, indice + 1)
        axe.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        # Plot also the training points
        axe.scatter(
            inputs[:, f1],
            inputs[:, f2],
            c=targets,
            edgecolors='k',
            cmap=plt.cm.Paired
        )
        plt.xlabel(iris_dataset.feature_names[f1])
        plt.ylabel(iris_dataset.feature_names[f2])

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

    fig.canvas.set_window_title(
        'Logistic Regression with two features for IRIS DATASET')
    plt.show()
