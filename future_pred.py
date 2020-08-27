import numpy as np
import pandas as pd
import yfinance as yf
import argparse
import datetime
import util

def predict(data, k, train_period, window_size=5, num_states=4, num_clusters=5, max_iter=5, latency=10):
    prev_hmm = None
    data = data[-train_period:]
    for t in range(latency, len(data)):
        print(t, len(data))
        temp_data = data[t - latency: t].values
        hmm = util.train_hmm(temp_data, k, window_size=window_size, num_states=num_states, num_clusters=num_clusters, max_iter=max_iter)
        prev_hmm = hmm
    state = prev_hmm.viterbi(data[-latency:])[-1]
    wm = (prev_hmm.weights[state] * prev_hmm.means[state, :, 0]).sum() / sum(hmm.weights[state])
    
    return wm

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", type=yf.Ticker)
    parser.add_argument("-tp", "--train-period", default=240, type=int)
    parser.add_argument("-l", "--latency", default=40, type=int)
    parser.add_argument("-s", "--states", default=4, type=int)
    parser.add_argument("-c", "--clusters", default=5, type=int)
    parser.add_argument("--max_iter", default=5, type=int)
    args = parser.parse_args()

    stock = args.stock.history(period="max")
    train_period = args.train_period
    latency = args.latency
    num_states = args.states
    num_clusters = args.clusters
    max_iter = args.max_iter
    
    data = util.gen_data(stock)[1]
    pred = predict(data, 4, train_period, num_states=num_states, num_clusters=num_clusters, max_iter=max_iter, latency=latency)
    print("Prediction for tomorrow: " + ("+" if pred >=0 else "-") + str(round(pred * 100, 2)))


