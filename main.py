import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import argparse

import sys

from hmm import HMM

def gen_data(stock):
    print(stock)
    fracChange = (stock.Close - stock.Open) / stock.Open
    fracHigh = (stock.High - stock.Open) / stock.Open
    fracLow = (stock.Open - stock.Low) / stock.Open
    train = pd.DataFrame(data=np.stack((fracChange.values, fracHigh.values, fracLow.values), axis=1), index=stock.index, columns=["fracChange", "fracHigh", "fracLow"])
    orig = pd.DataFrame(data=np.stack((stock.Open.values, stock.Close.values), axis=1), index=stock.index, columns=["Open", "Close"])
    return orig, train

def random_splits(n_splits, total):
    splits = np.zeros((n_splits))
    for i in range(n_splits - 1):
        splits[i] = total * np.random.rand()
        total -= splits[i]
    splits[-1] = total
    return splits

def train_hmm(X, k, prev_hmm=None, window_size=5, num_clusters=5, num_states=4, max_iter=5):
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
        weights = prev_hmm.weights_
        means = prev_hmm.means
        cov = prev_hmm.cov
    
    hmm = HMM(pi, A, weights, means, cov, num_states)
    hmm.em(X, window_size=3, max_iter=max_iter)
    
    return hmm

def predict(data, k, start, end, window_size=5, num_states=4, num_clusters=5, max_iter=5, latency=10):
    predictions = []
    prev_hmm = None
    for t in range(len(data.loc[:start]) - latency * 4, len(data.loc[:end])):
        print(t, len(data.loc[:end]))
        temp_data = data[t - latency: t].values 
        hmm = train_hmm(temp_data, k, window_size=window_size, num_states=num_states, num_clusters=num_clusters, max_iter=max_iter)
        if(t >= len(data.loc[:end]) - len(data.loc[start:end])):
            state = hmm.viterbi(temp_data)[-1]
            wm = (hmm.weights[state] * hmm.means[state, :, 0]).sum() / sum(hmm.weights[state])
            predictions.append(wm)
        prev_hmm = hmm
    return predictions

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(prog="stockhmm")
    parser.add_argument("stock", type=yf.Ticker)
    parser.add_argument("-r", "--range", required=True, nargs=2, type=pd.to_datetime)
    parser.add_argument("-l", "--latency", default=40, type=int)
    parser.add_argument("-s", "--states", default=4, type=int)
    parser.add_argument("-c", "--clusters", default=5, type=int)
    parser.add_argument("--max-iter", default=5, type=int)
    args = parser.parse_args()

    stock = args.stock.history(period="max")
    period = args.range
    latency = args.latency
    num_states = args.states
    num_clusters = args.clusters
    max_iter = args.max_iter

    orig, data = gen_data(stock)
    print(orig.loc[period[0]: period[1]][latency:])
    pred_change = predict(data, 4, period[0], period[1], num_states=num_states, num_clusters=num_clusters, max_iter=max_iter, latency=latency)
    pred_val = [orig.Open.loc[period[0]] * (1 + pred_change[0])]
    for i in range(1, len(pred_change)):
        pred_val.append(pred_val[-1] * (1 + pred_change[i]))

    df = pd.DataFrame(data={"Orig" : orig.Close.loc[period[0] : period[1]], "Pred" : pred_val})
    print(df)
    df.plot.line(y=["Orig", "Pred"])
    plt.show()
    
