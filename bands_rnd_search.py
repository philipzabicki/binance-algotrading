from gc import collect
from random import randint
from math import copysign
from csv import writer
from time import time
import numpy as np
import pandas as pd
from enviroments.BandParametrizerEnv import BandParametrizerEnv
from statistics import mean
from datetime import datetime
from dateutil.parser import parse
from multiprocessing import Pool, cpu_count
#from scipy.stats import skew, kurtosis

import get_data

SLIPPAGES = {'market_buy':(1.000022, 0.000045), 'market_sell':(0.999971, 0.000035), 'SL':(1.0, 0.000002)}

'''from pympler import summary, muppy
import pprint
import gc

from pympler import asizeof
def get_attributes_and_deep_sizes(obj):
    attributes_and_sizes = {}
    for attribute_name in dir(obj):
        attribute_value = getattr(obj, attribute_name)
        _size = asizeof.asizeof(attribute_value)
        if _size>1_000:
            attributes_and_sizes[attribute_name] = asizeof.asizeof(attribute_value)
    return attributes_and_sizes'''

# Calculates and returns linear regression slope but predictor variable(X) are natural numbers from 1 to len of dependent variable(Y)
# Y are supposed to be balance divided by initial balance ratios per every env step
def minutes_since(data_string):
    diff = datetime.now() - parse(data_string)
    minutes = diff.total_seconds() / 60
    return int(minutes)

def linear_reg_slope(Y):
    Y = np.array(Y)
    n = len(Y)
    X = np.arange(1, n+1)
    #print(f'X: {X}')
    x_mean = np.mean(X)
    Sxy = np.sum(X*Y)- n*x_mean*np.mean(Y)
    Sxx = np.sum(X*X)-n*x_mean**2
    return Sxy/Sxx

#def get_means():
    buy = pd.read_csv('slippages_market_buy.csv')
    sell = pd.read_csv('slippages_market_sell.csv')
    SL = pd.read_csv('slippages_StopLoss.csv')
    return {'market_buy':(buy.mean(), buy.std()), 'market_sell':(sell.mean(), sell.std()), 'SL':(SL.mean(), SL.std())}

'''def get_statistics():
    buy = pd.read_csv('market_buy_slippages.csv')
    sell = pd.read_csv('market_sell_slippages.csv')
    SL = pd.read_csv('SL_slippages.csv')

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
    }'''

def run_indefinitely(_):
    df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=999_999)
    df = df.drop(columns='Opened').to_numpy()
    df = np.hstack((df, np.zeros((df.shape[0], 1))))
    env = BandParametrizerEnv(df=df[-minutes_since('22-03-2023'):,:], init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPPAGES, visualize=False, Render_range=60, write_to_csv=False)
    episodes = randint(100, 2_500)
    timers = []
    results = []
    i, timer = 0, time()
    while len(results)<episodes:
        i+=1
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if reward!=0:
            _5 = len(info['trades_PNL_ratio'])//20
            slope_25_percentile = linear_reg_slope(info['trades_PNL_ratio'][-_5*5:])
            slope_50_percentile = linear_reg_slope(info['trades_PNL_ratio'][-_5*10:])
            slope_75_percentile = linear_reg_slope(info['trades_PNL_ratio'][-_5*15:])
            slope_95_percentile = linear_reg_slope(info['trades_PNL_ratio'][-_5*19:])
            slope_avg = (slope_75_percentile-slope_95_percentile)+(slope_50_percentile-slope_75_percentile)+(slope_25_percentile-slope_50_percentile)
            slope_avg = copysign(abs(slope_avg)**(1/3), slope_avg)
            #print(f'slope_avg {slope_avg} reward {reward}')
            if reward<0 and slope_avg<0: indicator = reward*slope_avg*-1
            else: indicator = reward*slope_avg
            if not np.isnan(indicator):
                #print(f'usd_gains: {info["gain"]:.2f}, indicator: {indicator:.4f}, order_count: {info["episode_orders"]} PNL_ratio: {info["pnl_ratio"]:.3f}, StDev: {info["stdev_pnl"]:.5f}, pos_hold_ratio: {info["position_hold_sums_ratio"]:.3f}, reg_slope_avg: {slope_avg} ')
                results.append([round(info["gain"],2), indicator, round(info["pnl_ratio"],3), round(info["stdev_pnl"],5), round(info["position_hold_sums_ratio"],3), \
                                slope_avg, round(action[0],4), int(action[1]), int(action[2]), int(action[3]), round(action[4],2)])
        _t = time()-timer
        #print(f'[{i}/{episodes}] {_t}')
        timers.append(_t)
        timer = time()
    print(f'MEAN EP TIME {mean(timers)}')
    ### Saving results to file
    results = sorted(results, key=lambda x: x[1])
    csv_filename = 'Z:/home/philipz_abicki/binance-algotrading/BTCTUSD1m_since0322_ATR.csv'
    with open(csv_filename, 'a', newline='') as file:
        _writer = writer(file)
        #writer.writerow(header)
        _writer.writerows(results)
    df = pd.read_csv(csv_filename)
    # Removing the first half of the rows
    df.sort_values(by=['indicator'], inplace=True)
    df = df.tail(df.shape[0] // 4)
    # Save sorted DataFrame back to the csv file
    df.to_csv(csv_filename, index=False)
    collect()

def main():
    # Number of processes is typically set to the number of available CPU cores
    num_processes = cpu_count()//2
    # Infinite loop to run the processes
    while True:
        with Pool(processes=1) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.map(run_indefinitely, range(num_processes))

#slippages = get_means()
#print(slippages)
if __name__=="__main__":
    df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=600)
    main()