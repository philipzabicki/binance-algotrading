from os import getcwd
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse
from scipy.stats import skew, kurtosis
from pympler import asizeof

def get_slips_stats():
    buy = pd.read_csv(getcwd()+'\settings\slippages_market_buy.csv')
    sell = pd.read_csv(getcwd()+'\settings\slippages_market_sell.csv')
    SL = pd.read_csv(getcwd()+'\settings\slippages_StopLoss.csv')
    return {'market_buy':(buy.mean(), buy.std()), 'market_sell':(sell.mean(), sell.std()), 'SL':(SL.mean(), SL.std())}

def get_slips_stats_advanced():
    buy = pd.read_csv(getcwd()+'\settings\slippages_market_buy.csv')
    sell = pd.read_csv(getcwd()+'\settings\slippages_market_buy.csv')
    SL = pd.read_csv(getcwd()+'\settings\slippages_market_buy.csv')
    return {
        'market_buy': {
            'mean': buy.mean(),
            'std': buy.std(),
            'skewness': buy.apply(skew),
            'kurtosis': buy.apply(kurtosis)
        },
        'market_sell': {
            'mean': sell.mean(),
            'std': sell.std(),
            'skewness': sell.apply(skew),
            'kurtosis': sell.apply(kurtosis)
        },
        'SL': {
            'mean': SL.mean(),
            'std': SL.std(),
            'skewness': SL.apply(skew),
            'kurtosis': SL.apply(kurtosis)
        }
    }

# Calculates and returns linear regression slope but predictor variable(X) are natural numbers from 1 to len of dependent variable(Y)
# Y are supposed to be balance divided by initial balance ratios per every env step
def linear_reg_slope(Y):
    Y = np.array(Y)
    n = len(Y)
    X = np.arange(1, n+1)
    #print(f'X: {X}')
    x_mean = np.mean(X)
    Sxy = np.sum(X*Y)- n*x_mean*np.mean(Y)
    Sxx = np.sum(X*X)-n*x_mean**2
    return Sxy/Sxx

def minutes_since(data_string):
    diff = dt.datetime.now() - parse(data_string)
    minutes = diff.total_seconds() / 60
    return int(minutes)

def get_attributes_and_deep_sizes(obj):
    attributes_and_sizes = {}
    for attribute_name in dir(obj):
        attribute_value = getattr(obj, attribute_name)
        _size = asizeof.asizeof(attribute_value)
        if _size>1_000:
            attributes_and_sizes[attribute_name] = asizeof.asizeof(attribute_value)
    return attributes_and_sizes