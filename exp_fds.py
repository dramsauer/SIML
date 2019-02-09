import lib.antcolony
import csv
import itertools
import math
import scipy
import numpy as np
from typing import AnyStr, Callable

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist, squareform

def read_columns_to_dict(path, d=","):
    """
    Write the correlation matrix into a dictionary of nodes.
    :path: fixed path to a csv file.
    :d: delimiter is set to , as default.
    :return: dictionary.
    """
    reader1, reader2 = itertools.tee(csv.reader(path, delimiter=d))
    columns = len(next(reader1))
    rows = len(next(reader2))
    counter_columns, counter_rows = 0, 0

    for i in reader1:
        counter_columns += 1
    for i in reader2:
        counter_rows += 1

    del reader1, reader2
    return counter_columns


def distcorr(X, Y):
    """
    Compute the distance correlation function on two random variables X and Y.
    :param X: Attribute column X.
    :param Y: Attribute column Y.
    :return: Distance correlation.
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor

def create_adjacency_matrix(path: list, distfkt: Callable) -> dict:
    """
    Creates an adjacency matrix for all columns of a dataset.
    :param path: Path to the desired csv file.
    :param distfkt: Distance function.
    :return: Adjacency matrix as dictionary of the form {"1_to_8": distcorr()}.
    """

    with open(path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    columns, rows = len(data_list[0]), len(data_list)
    adj_matrix = {}

    for i in range(0, columns):
        for j in range(0, columns):
            adj_matrix[str(i) + "_to_" + str(j)] = distfkt(data_list[i], data_list[j])

    return adj_matrix

print(create_adjacency_matrix("data/abalone.csv", distcorr))
