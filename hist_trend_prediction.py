import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import argparse
import util

def predict(data, k, start, end, pred_dist=0, window_size=5, num_states=4, num_clusters=5, max_iter=5, latency=10):
    predictions = []
    prev_hmm = None
    for t in range(len(data.loc[:start]) - latency * 4, len(data.loc[:end]) - pred_dist):
        print(t, len(data.loc[:end]))
        temp_data = data[t - latency: t].values 
        hmm = util.train_hmm(temp_data, k, prev_hmm=prev_hmm, window_size=window_size, num_states=num_states, num_clusters=num_clusters, max_iter=max_iter)
        if(t >= len(data.loc[:end]) - len(data.loc[start:end]) - pred_dist):
            state = hmm.viterbi(temp_data)[-1]
            wm = (hmm.weights[state] * hmm.means[state, :, 0]).sum() / sum(hmm.weights[state])
            predictions.append(wm)
        prev_hmm = hmm
    return predictions

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", type=yf.Ticker)
    parser.add_argument("-r", "--range", required=True, nargs=2, type=pd.to_datetime)
    parser.add_argument("-d", "--distance", default=0, type=int)
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
    pred_dist = args.distance

    orig, data = util.gen_data(stock)
    pred_change = predict(data, 4, period[0], period[1], pred_dist=pred_dist, num_states=num_states, num_clusters=num_clusters, max_iter=max_iter, latency=latency)
    pred_val = [orig.Open[np.where(orig.index == period[0])[0] - pred_dist].values[0] * (1 + pred_change[0])]
    
    for i in range(1, len(pred_change)):
        pred_val.append(pred_val[-1] * (1 + pred_change[i]))

    df = pd.DataFrame(data={"Orig" : orig.Close.loc[period[0] : period[1]], "Pred" : pred_val})
    print(df)
    df.plot.line(y=["Orig", "Pred"])
    plt.show()
    
