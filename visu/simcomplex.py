import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ripser import Rips, plot_dgms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import genfromtxt

def makeSparseDM(X, thresh):
    """
    Helper function to make a sparse distance matrix.
    :param X: Dataset to be processed.
    :param thresh: Treshold to be declined.
    :return: Sparse correlation distance matrix.
    """
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

def plot_vr_complex(path: str, delimiter: str = ",", thresh: float = 1.0,
                    maxdim: int = 3, coeff = 3, barcode: bool = True) -> np.ndarray:
    """
    Plots the Vietoris Rips complex and returns the data.
    :param path: Path to the desired csv file.
    :param delimiter: Delimiter for the csv file.
    :return: Data for a persistence diagram of a Vietoris Rips complex.
    """
    rips = Rips(maxdim = maxdim, coeff = coeff, do_cocycles = True)
    data = genfromtxt(path, delimiter=delimiter)
    diagrams = rips.fit_transform(data, distance_matrix=False)
    rips.plot(diagrams)
    return diagrams

plot_vr_complex('../../data/MOBISIG/USER1/SIGN_FOR_USER1_USER2_1.csv')