"""
This class contains common functions used for stock prediction and HMM training
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import argparse

from hmm import HMM

#######################
# Generate Stock Data #
#######################
def gen_data(stock):
    """Generates stock data

    Args:
        stock: a dataframe containing the stock data
    Returns:
        orig: the original open and close prices
        train: the training data containing return value, fraction of the high value and fraction of the low value
    """
    fracChange = (stock.Close - stock.Open) / stock.Open
    fracHigh = (stock.High - stock.Open) / stock.Open
    fracLow = (stock.Open - stock.Low) / stock.Open
    train = pd.DataFrame(data=np.stack((fracChange.values, fracHigh.values, fracLow.values), axis=1), index=stock.index, columns=["fracChange", "fracHigh", "fracLow"])
    orig = pd.DataFrame(data=np.stack((stock.Open.values, stock.Close.values), axis=1), index=stock.index, columns=["Open", "Close"])
    return orig, train

def random_splits(n_splits, total):
    """Creates a numpy array with values that represent random splits of a value

    Args:
        num_splits: the number of splits
        total: the value to be split
    Returns:
        an array of splits
    """
    splits = np.zeros((n_splits))
    for i in range(n_splits - 1):
        splits[i] = total * np.random.rand()
        total -= splits[i]
    splits[-1] = total
    return splits

#############
# Train HMM #
#############
def train_hmm(X, k, prev_hmm=None, window_size=5, num_clusters=5, num_states=4, max_iter=10):
    """Trains an HMM
    
    Args:
        X: the data
        k: the number of sequences to divide the data into
        prev_hmm: an HMM object to set the parameters
        window_size: the window size of exponential weighting
        num_clusters: the number of mixtures in each GMM
        num_states: the number of states in the HMM
        max_iter: the max number of iterations allowed in the HMM
    Returns:
        the trained HMM
    """
    temp_X = X.reshape((k, X.shape[0] // k, X.shape[1]))
    num_features = X.shape[1] 
    
    if(prev_hmm is None):
        pi = random_splits(num_states, 1)
        A = np.array([random_splits(num_states, 1) for _ in range(num_states)])
        weights = np.random.rand(num_states, num_clusters) / num_clusters
        means = np.random.rand(num_states, num_clusters, num_features) * .6 - .3
        cov = np.tile(np.eye(num_features), (num_states, num_clusters, 1, 1))
        for i in range(num_states):
            weights[i] = weights[i] / weights[i].sum()
    else:
        pi = prev_hmm.pi
        A = prev_hmm.A
        weights = prev_hmm.weights
        means = prev_hmm.means
        cov = prev_hmm.cov
    
    hmm = HMM(pi, A, weights, means, cov, num_states)
    hmm.em(X, window_size=3, max_iter=max_iter)
    
    return hmm
