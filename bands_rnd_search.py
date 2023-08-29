import numpy as np
import pandas as pd
from gc import collect
from os import getcwd 
from random import randint
from csv import writer
from time import time
from statistics import mean
from multiprocessing import Pool, cpu_count
import get_data
from enviroments.BandsStratEnv import BandsStratEnvSpot
from utility import minutes_since, get_slips_stats

CPU_CORES_COUNT = cpu_count()//2
CPU_CORES_COUNT = 6
#REPORT_FULL_PATH = 'Z:/home/philipz_abicki/binance-algotrading/reports/BTCTUSD1m_since0322_ATR.csv'
REPORT_FULL_PATH = getcwd()+'/reports/BTCTUSD1m_since0322_ATR.csv'
EPISODES = randint(10, 250)
TICKER, ITV, FUTURES, START_DATE = 'BTCTUSD', '1m', False, '22-03-2023'

def run_indefinitely(_):
    df = get_data.by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=3_600)
    df = df.drop(columns='Opened').to_numpy()
    df = np.hstack((df, np.zeros((df.shape[0], 1))))
    env = BandsStratEnvSpot(df=df[-minutes_since(START_DATE):,:], init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=get_slips_stats(), visualize=False, Render_range=60, write_to_csv=False)

    timers, results = [], []
    i, timer = 0, time()
    while len(results)<EPISODES:
        i+=1
        action = env.action_space.sample()
        _, reward, _, info = env.step(action)
        if reward!=0 and not np.isnan(reward):
            #print(f'usd_gains: {info["gain"]:.2f}, indicator: {indicator:.4f}, order_count: {info["episode_orders"]} PNL_ratio: {info["pnl_ratio"]:.3f}, StDev: {info["stdev_pnl"]:.5f}, pos_hold_ratio: {info["position_hold_sums_ratio"]:.3f}, reg_slope_avg: {slope_avg} ')
            results.append([round(info["gain"],2), round(reward,4), round(info["pnl_ratio"],3), round(info["hold_time_ratio"],3), round(info["avg_trades_ratio"],3),
                            round(info['slope_indicator'],4), round(action[0],4), int(action[1]), int(action[2]), int(action[3]), round(action[4],2)])
        _t = time()-timer
        #print(f'[{i}/{episodes}] {_t}')
        timers.append(_t)
        timer = time()
    print(f'MEAN EP TIME {mean(timers)}')
    ### Saving results to file
    #results = sorted(results, key=lambda x: x[1])
    with open(REPORT_FULL_PATH, 'a', newline='') as file:
        _writer = writer(file)
        #writer.writerow(header)
        _writer.writerows(results)
    df = pd.read_csv(REPORT_FULL_PATH)
    # Removing the first half of the rows
    df.sort_values(by=['reward'], inplace=True)
    #df = df.tail(df.shape[0] // 1)
    # Save sorted DataFrame back to the csv file
    df.to_csv(REPORT_FULL_PATH, index=False)
    collect()

def main():
    df = get_data.by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=0)
    # Infinite loop to run the processes
    while True:
        with Pool(processes=CPU_CORES_COUNT) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.map(run_indefinitely, range(CPU_CORES_COUNT))

if __name__=="__main__":
    main()