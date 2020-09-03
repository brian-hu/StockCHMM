"""
This class contains the HMM model
"""

import numpy as np

class HMM:
    def __init__(self, pi, A, weights, means, cov, num_states):
        """Initialize the HMM with the required parameters

        Args:
            pi: the initial distribution
            A: the transition probabilities
            weights: the weights of the GMMs
            means: the means of the GMMs
            cov: the covarances of the GMMs
            num_states: the number of states in the HMM
        """

        self.pi = pi
        self.A = A
        self.weights = weights
        self.means = means
        self.cov = cov
        self.num_clusters = len(weights[0])
        self.num_states = num_states
    
    #############################
    # Multivariate Gaussian PDF #
    #############################
    def norm_pdf(self, x, mu, sigma):
        """Finds the pdf of x given the mean and covariance of a multivariate gaussian function

        Note that this function can find the pdf of many vectors if x is a matrix

        Args:
            x: the vector(s)
            mu: the mean of the MVG
            sigma: the covariance of the MVG
        Returns:
            A vector of pdfs
        """
        size = len(x) if x.ndim == 1 else x.shape[1]
        if(size == len(mu) and (size, size) == sigma.shape):
            det = np.linalg.det(sigma)
            denom = (np.power((2 * np.pi), float(size) / 2) * np.sqrt(abs(det)))
            x_mu = np.array(x - mu)
            inv = np.linalg.inv(sigma)
            numer = ((x_mu.dot(inv)) * x_mu).sum(axis=1)
            exp = np.exp(-.5 * numer)
            return exp / denom

    ################################### 
    # Calculate Exponential Weighting #
    ###################################
    def calc_ema(self, T, window_size):
        """Calculates the weights of the data based on the window size

        Args:
            T: the length of the data
            window_size: the window_size for weighting
        Returns:
            A vector containing the weights
        """
        n = np.zeros((T))
        p = 2 / (1 + window_size)
        for t in range(T):
            if(t < window_size):
                n[t] = np.power(1 - p, T - window_size) / window_size
            elif(window_size <= t < T - 1):
                n[t] = p * np.power(1 - p, T - t - 1)
            else:
                n[t] = p
        return n

    ############################ 
    # Calculate Log Likelihood #
    ############################
    def log_prob(self, c):
        return float("-inf") if any(c == 0) else np.log(c).sum()

    ####################################### 
    # Calculate Observation Probabilities #
    #######################################
    def calc_B(self, X):
        """Calculates observation probabilities

        Args:
            X: the data
        Returns:
            B: Matrix of observation probabilities
            B2: Matrix of observation probabilities that also considers state
        """
        B = np.zeros((X.shape[0], self.num_states))
        B2 = np.zeros((X.shape[0], self.num_states, self.num_clusters))
       
        for state in range(self.num_states):
            for cluster in range(self.num_clusters):
                B2[:,state, cluster] = self.norm_pdf(X, self.means[state, cluster], self.cov[state, cluster])
        
        for t in range(X.shape[0]):
            B[t] = (self.weights * B2[t]).sum(axis=1)
        return B, B2

    ######################
    # Normalize a vector #
    ######################
    def normalize(self, x):
        """Makes a vector sum to 1

        Args:
            x: the vector
        Returns:
            the normalized vector
        """
        c = x.sum()
        s = 1 if c == 0 else c
        M = x / s
        return M, c

    ################### 
    # Make Stochastic #
    ###################
    def mk_stochastic(self, x):
        """Makes a matrix stochastic (every row sums to 1)

        Args:
            x: the matrix/vector
        Returns:
            the stochastic matrix
        """
        if(x.ndim == 1):
            return self.normalize(x)[0]
        elif(x.ndim == 2):
            s = x.sum(axis=1)
            s[s == 0] = 1
            return (x.T / s).T

    ##################### 
    # Forward Algorithm #
    #####################
    def forward_algorithm(self, X, B):
        """Calculates the forward probabilities for the EM algorithm

        Args:
            X: the data
            B: the observation probabilities
        Returns:
            alpha: the forward probabilities
            loglik: the log likelihood
        """
        alpha = np.zeros((X.shape[0], self.num_states))
        scale = np.zeros(X.shape[0])
        
        alpha[0] = self.pi * B[0]
        alpha[0], scale[0] = self.normalize(alpha[0])
        for t in range(1, X.shape[0]):
            alpha[t] = self.A.dot(alpha[t - 1]) * B[t]
            alpha[t], scale[t] = self.normalize(alpha[t]) 

        loglik = self.log_prob(scale)

        return (alpha, loglik)
   
    ######################
    # Backward Algorithm #
    ######################
    def backward_algorithm(self, alpha, X, B, B2, ema):
        """Calculates the backward probabilities, gamma, xi and gamma2

        Args:
            alpha: the forward probabilities
            X: the data
            B: the observation probabilities
            B2: the observation probabilities depending on state
            ema: the weights
        Returns:
            beta: the backwards probabilities
            gamma: the probability of being in state i at time t
            gamma2: the probability of being in state i at time t and state j at time t + 1
            xi_sum: 
        """
        beta = np.zeros(alpha.shape)
        gamma = np.zeros(alpha.shape)
        xi_sum = np.zeros((self.num_states, self.num_states))
        gamma2 = np.zeros((X.shape[0], self.num_states, self.num_clusters))
        
        #initialize t=T values
        beta[-1] = np.ones((self.num_states))
        gamma[-1] = self.normalize(ema[-1] * alpha[-1] * beta[-1])[0]
        denom = B[-1]
        denom[denom == 0] = 1
        gamma2[-1] = ema[-1] * B2[-1] * self.weights * np.tile(gamma[-1], (self.num_clusters, 1)).T / np.tile(denom, (self.num_clusters, 1)).T
        for t in range(X.shape[0] - 2, -1, -1):
            b = beta[t + 1] * B[t + 1]
            #beta
            beta[t] = self.normalize(self.A.dot(b))[0]
            #gamma
            gamma[t] = self.normalize(alpha[t] * beta[t])[0]
            #xi_sum
            xi_sum = (xi_sum + self.normalize((self.A * np.outer(alpha[t], b)))[0]) * ema[t]
            #gamma2 
            denom = B[t]
            denom[denom == 0] = 1
            gamma2[t] = ema[t] * B2[t] * self.weights * np.tile(gamma[t], (self.num_clusters, 1)).T / np.tile(denom, (self.num_clusters, 1)).T
        return beta, gamma, gamma2, xi_sum
    
    def fwdback(self, X, B, B2, ema):
        """Combines the forward and backward algorithm and returns all the appropriate values"""
        alpha, loglik = self.forward_algorithm(X, B)
        beta, gamma, gamma2, xi_sum  = self.backward_algorithm(alpha, X, B, B2, ema)
        return (alpha, beta, gamma, gamma2, xi_sum, loglik)
    
    #####################
    # Viterbi Algorithm #
    #####################
    def viterbi(self, X):
        """Finds the most probable state sequence given the current parameters

        Args:
            X: the data
        Returns:
            the state sequence
        """
        delta = np.zeros((X.shape[0], self.num_states))
        prev_states = np.zeros((X.shape[0], self.num_states)) 
        path = np.zeros((X.shape[0]), dtype="int64")
        B = self.calc_B(X)[0]
        scale = np.zeros((X.shape[0]))
        
        delta[0], scale[0] = self.normalize(self.pi * B[0])
        for t in range(1, X.shape[0]):
            for j in range(self.num_states):
                delta[t, j] = np.max(self.A[:, j] * delta[t - 1] * B[t, j])
                prev_states[t, j] = np.argmax(self.A[:, j] * delta[t - 1])
            delta[t], scale[t] = self.normalize(delta[t])
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(X.shape[0] - 2, -1, -1):
            temp = int(path[t + 1])
            path[t] = prev_states[t + 1, temp]
        return path
    
    #####################
    # Update Parameters #
    #####################
    def update(self, postmix, m, op, exp_num_trans, exp_num_visits):
        """Updates all the parameters

        Args:
            postmix: values used to update the weights
            m: sum of weighted observations for eah state
            op: outer product of the weighted observations with the normal observations
            exp_num_trans: values used to update the transition probabilities
            exp_num_visits: values used to update the initial probabilities
        """
        #update pi
        self.pi = self.mk_stochastic(exp_num_visits)
        #update A
        self.A = self.mk_stochastic(exp_num_trans)
        #update weights
        self.weights = self.mk_stochastic(postmix)
        #update means
        for i in range(self.num_states):
            self.means[i] = (m[i].T / postmix[i]).T
        #update cov
        for i in range(self.num_states):
            for j in range(self.num_clusters):
                self.cov[i, j] = op[i, j] / postmix[i, j] - np.outer(self.means[i, j], self.means[i, j]) + 0.01 * np.eye(op.shape[2])

    #################### 
    # Calculate Values #
    ####################
    def calc_quant(self, X, window_size):
        """Calculates the necessary values to update the model

        Args:
            X: data
            window_size: the window_size for exponential weighting
        Returns:
            loglik: the sum of log likelihoods of all the time-series
            postmix: values used to update the weights
            m: sum of weighted observations for eah state
            op: outer product of the weighted observations with the normal observations
            exp_num_trans: values used to update the transition probabilities
            exp_num_visits: values used to update the initial probabilities
        """
        numex = X.shape[0] if X.ndim == 3 else 1
        exp_num_trans = np.zeros((self.num_states, self.num_states))
        exp_num_visits = np.zeros((self.num_states))
        postmix = np.zeros((self.num_states, self.num_clusters))
        ema = self.calc_ema(X.shape[1] if X.ndim == 3 else X.shape[0], window_size)
        loglik_total = 0
        for ex in range(numex):
            temp_X = X[ex] if numex > 1 else X
            B, B2 = self.calc_B(temp_X)
            alpha, beta, gamma, gamma2, xi_sum, loglik = self.fwdback(temp_X, B, B2, ema)
            exp_num_trans = exp_num_trans + xi_sum
            exp_num_visits = gamma[0]
            postmix = postmix + gamma2.sum(axis=0)
            loglik_total += loglik
        
            m = np.zeros((self.num_states, self.num_clusters, temp_X.shape[1]))
            op = np.zeros((self.num_states, self.num_clusters, temp_X.shape[1], temp_X.shape[1]))
            for state in range(self.num_states):
                for cluster in range(self.num_clusters):
                    w = gamma2[:, state, cluster]
                    wx = (temp_X.T * w).T
                    m[state, cluster] = wx.sum(axis=0)
                    op[state, cluster] = wx.T.dot(temp_X)
        return loglik, exp_num_trans, exp_num_visits, postmix, m, op
    
    ################
    # EM Algorithm #
    ################
    def em(self, X, window_size=5, max_iter=10, tol=0.1):
        """Fits the model with data using the EM algorithm

        Args:
            X: the data
            window_size: window size for exponential weighting
            max_iter: the max number of EM updates allowed
            tol: the tolerance level to automatically stop the updates
        """
        self.X = X
        count = 0
        prev_loglik = float("-inf")
        logs = []
        while(count < max_iter):
            loglik, exp_num_trans, exp_num_visits, postmix, m, op = self.calc_quant(X, window_size)
            self.update(postmix, m, op, exp_num_trans, exp_num_visits)
            count += 1
            logs.append(loglik)
            if(abs(prev_loglik - loglik) < tol):
                break
            else:
                prev_loglik = loglik
