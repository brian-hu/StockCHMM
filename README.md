# Stock CHMM
This project uses a continuous Hidden Markov Model to predict stock prices. It has two main functionalities: historical range comparison and predicting tomorrow's stock price.

## How It Works
A Hidden Markov Model is a model that assumes an observed sequence is controlled by a finite number of "hidden" states. The current state is assumed to produce an output, as well as affect the next hidden state. It has three parameters:
* initial distribution: the probability of each state to be the initial state
* transition probabilities: the probability to transition from one state to another state
* observation distribution: the probability of observing an output given a hidden state

In cases with continous outputs (such as in this project), observation distribution is modelled using statistical distributions such as a Gaussian Mixture Model (what we used in this project). Given an observed sequence, the model updates and improves its parameters using the EM algorithm.

**Improvements we made:**
* The model trains on a sliding window, so that the most recent data is most impactful.
* The model weights recent data more heavily so that the model can match rapid changes of stock prices more quickly

## Historical Trend Prediction
To run historical trend prediction, run hist_trend_pred.py using python3. It has the following parameters:
* stock (required) - the stock symbol/ticker (i.e. AAPL for Apple)
* --range (required) - the range of prediction
* --distance - the number of days into the future to predict (i.e. distance = 5 will predict 1/5/2020's closing stock price on 1/1/2020), default = 0
* --latency - the number of days in the training window, default = 20
* --states - the number of states in the HMM, default =  4
* --clusters - the number of mixtures used in the Gaussian Mixture Models used as the output distribution, default = 5
Use --help for more details

## Future Price Prediction
To run future price prediction, run future_pred.py using python3. It has the following parameters:
* stock (required) - the stock symbol/ticker (i.e. AAPL for Apple)
* --train-period - the number of days used to train the model, default = 252
* --latency - the number of days in the training window, default = 20
* --states - the number of states in the HMM, default = 4
* --clusters - the number of mixtures used in the Gaussian Mixture Models used as the output distribution, default = 5
Use --help for more details

## Examples
Example of historical range prediction:
```
python3 hist_trend_pred.py SPY --range 1/2/2015 1/2/2016 --distance 0 --latency 20 --states 4 --clusters 5
```
Example of future price prediction:
```
python3 future_price_pred.py SPY --train-period 252 --latency 20 --states 4 --clusters 5
```
