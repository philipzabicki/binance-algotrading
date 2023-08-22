from gc import collect
from random import randint
from math import copysign
from csv import writer
from time import time
import numpy as np
import pandas as pd
from enviroments.BandParametrizerEnv import BandParametrizerEnv
from statistics import mean
from multiprocessing import Pool, cpu_count

import get_data
from utility import minutes_since, linear_reg_slope, get_slips_stats

CPU_CORES_COUNT = cpu_count()//2
SLIPPAGES = get_slips_stats()
REPORT_FULL_PATH = 'Z:/home/philipz_abicki/binance-algotrading/reports/BTCTUSD1m_since0322_ATR.csv'
EPISODES = randint(100, 2_500)
TICKER, ITV, FUTURES, START_DATE = 'BTCTUSD', '1m', False, '22-03-2023'

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

def run_indefinitely(_):
    df = get_data.by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=3_600)
    df = df.drop(columns='Opened').to_numpy()
    df = np.hstack((df, np.zeros((df.shape[0], 1))))
    env = BandParametrizerEnv(df=df[-minutes_since(START_DATE):,:], init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPPAGES, visualize=False, Render_range=60, write_to_csv=False)

    timers, results = [], []
    i, timer = 0, time()
    while len(results)<EPISODES:
        i+=1
        action = env.action_space.sample()
        _, reward, _, info = env.step(action)
        if reward!=0 and not np.isnan(reward):
            #print(f'usd_gains: {info["gain"]:.2f}, indicator: {indicator:.4f}, order_count: {info["episode_orders"]} PNL_ratio: {info["pnl_ratio"]:.3f}, StDev: {info["stdev_pnl"]:.5f}, pos_hold_ratio: {info["position_hold_sums_ratio"]:.3f}, reg_slope_avg: {slope_avg} ')
            results.append([round(info["gain"],2), reward, round(info["pnl_ratio"],3), round(info["stdev_pnl"],5), round(info["position_hold_sums_ratio"],3),
                            round(info['slope_indicator'],4), round(action[0],4), int(action[1]), int(action[2]), int(action[3]), round(action[4],2)])
        _t = time()-timer
        #print(f'[{i}/{episodes}] {_t}')
        timers.append(_t)
        timer = time()
    print(f'MEAN EP TIME {mean(timers)}')
    ### Saving results to file
    results = sorted(results, key=lambda x: x[1])
    with open(REPORT_FULL_PATH, 'a', newline='') as file:
        _writer = writer(file)
        #writer.writerow(header)
        _writer.writerows(results)
    df = pd.read_csv(REPORT_FULL_PATH)
    # Removing the first half of the rows
    df.sort_values(by=['indicator'], inplace=True)
    df = df.tail(df.shape[0] // 4)
    # Save sorted DataFrame back to the csv file
    df.to_csv(REPORT_FULL_PATH, index=False)
    collect()

def main():
    # Infinite loop to run the processes
    while True:
        with Pool(processes=1) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.map(run_indefinitely, range(CPU_CORES_COUNT))

if __name__=="__main__":
    main()