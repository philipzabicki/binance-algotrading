from numpy import isnan, hstack, zeros, inf
from pandas import read_csv
# import pandas as pd
from gc import collect
from os import getcwd
from random import randint
from csv import writer
from time import time, sleep
from statistics import mean
from multiprocessing import Pool, cpu_count
from get_data import by_DataClient, by_BinanceVision
from enviroments.macd import MACDStratEnv
from utility import minutes_since, seconds_since, get_market_slips_stats
from definitions import REPORT_DIR
import cProfile

CPU_CORES_COUNT = 1  # cpu_count()
EPISODES_PER_CORE = 100
TICKER, ITV, M_TYPE, START_DATE = 'BTCFDUSD', '1m', 'spot', '11-09-2023'
ENVIRONMENT = MACDStratEnv
SLIPP = get_market_slips_stats()
REPORT_FULL_PATH = REPORT_DIR + f'{TICKER}{M_TYPE}{ITV}since{START_DATE}.csv'


def run_indefinitely(_, df):
    profiler = cProfile.Profile()
    profiler.enable()

    env = ENVIRONMENT(df=df, no_action_finish=inf,
                      init_balance=1_000, fee=0.0, coin_step=0.00001,
                      slippage=SLIPP, visualize=False, verbose=False, render_range=60)
    timers, results = [], []
    i, timer = 0, time()
    while len(results) < EPISODES_PER_CORE:
        i += 1
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        # print(obs, reward, done, info)
        if not isnan(reward):
            row = [v for k, v in env.exec_env.info.items()] + [a for a in action]
            results.append(row)
        _t = time() - timer
        # print(f'[{i}/{EPISODES_PER_CORE}] {_t}')
        timers.append(_t)
        timer = time()
    print(f'MEAN EP TIME {mean(timers)}')
    ### Saving results to file
    # results = sorted(results, key=lambda x: x[1])
    with open(REPORT_FULL_PATH, 'a', newline='') as file:
        _writer = writer(file)
        # writer.writerow(header)
        _writer.writerows(results)
    profiler.disable()
    profiler.print_stats(sort='tottime')


def main():
    # Infinite loop to run the processes
    while 1:
        start_t = time()
        # df = by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=3_600)
        df = by_BinanceVision(ticker=TICKER, interval=ITV, type=M_TYPE, data='klines', delay=129_600)
        df = df.drop(columns='Opened').to_numpy()[-minutes_since(START_DATE):, :]
        df = hstack((df, zeros((df.shape[0], 1))))
        # df = df[-265_000:,:]
        with Pool(processes=CPU_CORES_COUNT) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.starmap(run_indefinitely, [(i, df) for i in range(CPU_CORES_COUNT)])
            # pool.map(run_indefinitely, range(CPU_CORES_COUNT))
        df = read_csv(REPORT_FULL_PATH)
        df.sort_values(df.columns[0], inplace=True)
        df.to_csv(REPORT_FULL_PATH, index=False)
        exec_time = time() - start_t
        eps = (CPU_CORES_COUNT * EPISODES_PER_CORE) / exec_time
        print(f'POOL exec time: {exec_time:.2f}s EpisodesPerSecond: {eps:.3f}')
        # sleep(10000)
        break
        collect()


if __name__ == "__main__":
    main()
