#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:03:53 2023

@author: brandonquach
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
from scipy.stats import kurtosis, skew, moment
from tqdm import tqdm

def pull_data(ticker, interval='1d'):
    '''
    Pulls the data through yahoo finance and calculates log returns. Returns 
    max history.
    
    Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    '''
    api = yf.Ticker(ticker)
 
    # Close here is the adjusted close price (after subtracting dividends when announced and stock splits)
    data = api.history(period='max',  interval=interval)
    data['prev_close'] = data['Close'].shift(1)
    data['log_return'] = np.log(data['Close'] / data['prev_close'])
    return data


def log_likelihood(params, log_returns):
    '''
    Derived log likelihood (taking out the normalization factor) for a merton 
    jump model that only allows a maximum of one jump per period.
    
    params = [mu, sigma, jump_rate, jump_mean, jump_std]
    
    Standard Brownian Motion Component ~ N(mu, sigma)
    Poisson Jump Variable ~ Poisson(jump_rate) (jump_rate assumed to be small)
    Jump Size ~ N(jump_mean, jump_std)
    '''
    mu, sigma, jump_rate, jump_mean, jump_std = params
    T = len(log_returns)
    # Drift Portion
    log_likelihood = 1 / sigma * np.exp(-((log_returns - (mu - (sigma ** 2) / 2)) ** 2) / 2 / (sigma ** 2))
    # Jump Portion
    log_likelihood += 1 / np.sqrt(sigma ** 2 + jump_std ** 2) * np.exp(-((log_returns - (mu - (sigma ** 2) / 2 + jump_mean)) ** 2) / 2 / (sigma ** 2 + jump_std ** 2))
    log_likelihood = np.log(log_likelihood) - jump_rate * T - T * np.log(np.sqrt(2 * np.pi))
    # Return sum of likelihoods over all observations
    return np.sum(log_likelihood)

def simple_mse_calibration(params, log_returns, n_sim=1000):
    '''
    Generates n_sim simulation draws using the given parameters and calculates 
    the mean squared error between sampled returns and the empirical returns.
    This method is exremely prone to local minimum and computationally 
    inefficient as it requires a large number of simulation draws to accurately
    calculate the mse. 
    '''
    mu, sigma, jump_rate, jump_mean, jump_std = params
    n = len(log_returns)
    drift = np.random.normal(mu, sigma, (n_sim, n))
    jump = np.random.normal(jump_mean, jump_std, (n_sim, n)) * np.random.poisson(jump_rate, (n_sim, n))
    predicted_log_returns = drift + jump
    error = ((predicted_log_returns - log_returns) ** 2).sum(axis=1).mean()
    return error

def method_of_moments(params, log_returns, n_sim=100000):
    '''
    Generates n_sim simulation draws of log returns. It provides the squared 
    error between the empirical estimates of the first five moments of the 
    distribution and the sampled estimates of the first five moments. Also 
    prone to local minimum with unintuitive results. 
    '''
    mu, sigma, jump_rate, jump_mean, jump_std = params
    drift = np.random.normal(mu, sigma, n_sim)
    jump = np.random.normal(jump_mean, jump_std, n_sim) * np.random.poisson(jump_rate, n_sim)
    predicted_log_returns = drift + jump
    
    error = 0
    
    # Moment match
    for k in range(5):
        error += (moment(predicted_log_returns, moment=k) - moment(log_returns, moment=k)) ** 2
    return error
    

def neg_log_likelihood(params, returns):
    '''
    Wrapper function to be passed to scipy minimize
    '''
    return -log_likelihood(params, returns)


def optimize_parameters(returns, trials=1000, bounds=None, mode='likelihood'):
    '''
    Loops through trials (default 100) times of random initializations of 
    the parameters within certain bounds made using assumptions to avoid
    falling into local maxima that this non-convex likelihood functino is 
    subject to.
    
    The assumptions are as follows
    
    1. Mean log return must be between -1 + 1
    2. Variances for both jump and drift must be positive
    3. There is a small probability of jumping per period in this case our
    bounds assume a maximum of 1 jump every 5 periods and minimum of 1 jump
    per 100 periods (though this can be tuned as well)
    4. Our jump mean is also somewhere been 0 and 2
    
    '''
    assert mode in ['likelihood', 'method_of_moments', 'mse']
    
    # Bounds must be in order of mu, sigma, jump_rate, jump_mean, jump_std
    mu_bound = abs(np.mean(returns))
    if bounds is None:
        bounds = ((-mu_bound, mu_bound), (1e-3, np.std(returns)), (1e-5, 0.5), (-1, 1), (1e-2, 1))
    
    
    best_params = None
    best_neg_likelihood = np.inf
    
    if mode == 'likelihood':
        calibration_func = neg_log_likelihood
        print('Using Likelihood')
    elif mode == 'method_of_moments':
        calibration_func = method_of_moments
        print('Using Method Of Moments')
    else:
        calibration_func = simple_mse_calibration 
        print('Using Simple MSE')
    
    for i in tqdm(range(trials)):
        initial_params = []
        # Generate uniformly a initialization between the bounds
        for l, r in bounds:
            # Cap out at 5 for variances which is definitely reasonable upper 
            # bound given it represents variation of stock returns
            if r == np.inf:
                r = 5
            initial_params.append(np.random.uniform(l, r))
        # estimate the parameters estimating MLE
        results = minimize(calibration_func, initial_params, args=(returns,), bounds = bounds, tol=1e-16)
        
        # Store if best
        mu, sigma, jump_rate, jump_mean, jump_std = results.x
        neg_likelihood = calibration_func(results.x, returns)
        if neg_likelihood <= best_neg_likelihood:
            best_neg_likelihood = neg_likelihood
            best_params = results.x
    
    # Return the best parameters found given the data.
    mu, sigma, jump_rate, jump_mean, jump_std = best_params
    return mu, sigma, jump_rate, jump_mean, jump_std 

def simulate_log_returns(mu, sigma, jump_rate, jump_mean, jump_std, n=1000):
    '''
    Generates outcomes using our estimates of each parameter of our distribution.
    Returns nominal returns instead of log returns.
    
    Log_Return = N(mu, sigma) + Poisson(jump_rate) * N(jump_mean, jump_std)
    '''
    drift = np.random.normal(mu, sigma, n)
    jump = np.random.normal(jump_mean, jump_std, n) * np.random.poisson(jump_rate, n)
    predicted_log_returns = drift + jump
    predicted_returns = np.exp(predicted_log_returns) - 1
    return predicted_returns

def returns_to_prices(start_price, predicted_returns, n=1000):
    '''
    Converts returns to prices given a start price
    '''
    prices = start_price * np.cumprod(1 + predicted_returns)
    return prices

