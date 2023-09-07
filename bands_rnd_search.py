from numpy import isnan, hstack, zeros
from pandas import read_csv
#import pandas as pd
from gc import collect
from os import getcwd 
from random import randint
from csv import writer
from time import time, sleep
from statistics import mean
from multiprocessing import Pool, cpu_count
from get_data import by_DataClient
from enviroments.BandsStratEnv import BandsStratEnv
from utility import minutes_since, get_slips_stats
import cProfile

#CPU_CORES_COUNT = cpu_count()
CPU_CORES_COUNT = 1
EPISODES_PER_CORE = 100
#CPU_CORES_COUNT = 6
#REPORT_FULL_PATH = 'Z:/home/philipz_abicki/binance-algotrading/reports/BTCTUSD1m_since0322_ATR.csv'
REPORT_FULL_PATH = getcwd()+'/reports/BTCTUSD1m_since0322_ATR.csv'
#EPISODES = randint(10, 250)
TICKER, ITV, FUTURES, START_DATE = 'BTCTUSD', '1m', False, '22-03-2023'

def run_indefinitely(_, df):
    profiler = cProfile.Profile() 
    profiler.enable()
    env = BandsStratEnv(df=df, 
                        init_balance=1_000, fee=0.00075, coin_step=0.00001, slippage=get_slips_stats(),
                        visualize=False, Render_range=60)
    timers, results = [], []
    i, timer = 0, time()
    while len(results)<EPISODES_PER_CORE:
        i+=1
        action = env.action_space.sample()
        _, reward, _, info = env.step(action)
        #print(obs, reward, done, info)
        if reward!=0 and not isnan(reward):
            #print(f'usd_gains: {info["gain"]:.2f}, indicator: {indicator:.4f}, order_count: {info["episode_orders"]} PNL_ratio: {info["pnl_ratio"]:.3f}, StDev: {info["stdev_pnl"]:.5f}, pos_hold_ratio: {info["position_hold_sums_ratio"]:.3f}, reg_slope_avg: {slope_avg} ')
            results.append([round(info["gain"],2), round(reward,4), round(info["PL_ratio"],3), round(info["PL_count_mean"],3), round(info["hold_ratio"],3),
                            round(info['slope_indicator'],4), round(action[0],4), round(action[1],3), round(action[2],3), int(action[3]), int(action[4]), int(action[5]), round(action[6],3)])
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
    profiler.disable()  # Zakończ profilowanie
    profiler.print_stats(sort='tottime')

def main():
    # Infinite loop to run the processes
    while 1:
        start_t = time()
        df = by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=3_600)
        df = df.drop(columns='Opened').to_numpy()
        df = hstack((df, zeros((df.shape[0], 1))))
        df = df[-minutes_since(START_DATE):,:]
        with Pool(processes=CPU_CORES_COUNT) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.starmap(run_indefinitely, [(i, df) for i in range(CPU_CORES_COUNT)])
            #pool.map(run_indefinitely, range(CPU_CORES_COUNT))
        df = read_csv(REPORT_FULL_PATH)
        df.sort_values(by=['reward'], inplace=True)
        df.to_csv(REPORT_FULL_PATH, index=False)
        exec_time = time()-start_t
        eps = (CPU_CORES_COUNT*EPISODES_PER_CORE)/exec_time
        print(f'POOL exec time: {exec_time:.2f}s EpisodesPerSecond: {eps:.3f}')
        #sleep(10000)
        break
        collect()

if __name__=="__main__":
    main()