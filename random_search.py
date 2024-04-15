import cProfile
from csv import writer
# import pandas as pd
from multiprocessing import Pool, cpu_count
from statistics import mean
from time import time

from numpy import isnan, inf
from pandas import read_csv

from definitions import REPORT_DIR
from enviroments import MACDOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision

# from utils.utility import get_slippage_stats

# CPU_CORES_COUNT = cpu_count()
CPU_CORES_COUNT = 1
EPISODES_PER_CORE = 256
TICKER, ITV, MARKET_TYPE, DATA_TYPE = 'BTCUSDT', '5m', 'um', 'klines'
TRADE_START_DATE = '2021-09-13'
TRADE_END_DATE = '2022-01-11'
DF_START_DATE = '2021-03-13'
DF_END_DATE = '2022-01-12'
ENVIRONMENT = MACDOptimizeSavingFuturesEnv
# SLIPP = get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market')
REPORT_FULL_PATH = REPORT_DIR + f'{TICKER}{MARKET_TYPE}{ITV}since{TRADE_START_DATE}.csv'


def run_indefinitely(_, df, df_mark=None):
    profiler = cProfile.Profile()
    profiler.enable()

    env = ENVIRONMENT(df=df, df_mark=df_mark, no_action_finish=inf,
                      start_date=TRADE_START_DATE, end_date=TRADE_END_DATE,
                      init_balance=1_000, fee=0.0005, coin_step=0.001,
                      # slippage=SLIPP,
                      visualize=False, verbose=False, render_range=60)
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
    profiler.print_stats(sort='cumtime')


def main():
    # Infinite loop to run the processes
    while 1:
        start_t = time()
        # df = by_DataClient(ticker=TICKER, interval=ITV, futures=FUTURES, statements=True, delay=3_600)
        df = by_BinanceVision(ticker=TICKER,
                              interval=ITV,
                              market_type=MARKET_TYPE,
                              data_type=DATA_TYPE,
                              start_date=DF_START_DATE,
                              end_date=DF_END_DATE,
                              split=False,
                              delay=345_600)
        df_mark = by_BinanceVision(ticker=TICKER,
                                   interval=ITV,
                                   market_type=MARKET_TYPE,
                                   data_type='markPriceKlines',
                                   start_date=DF_START_DATE,
                                   end_date=DF_END_DATE,
                                   split=False,
                                   delay=345_600)
        print(f'df {df}')
        with Pool(processes=CPU_CORES_COUNT) as pool:
            # Each process will call 'run_indefinitely_process'
            # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
            pool.starmap(run_indefinitely, [(i, df, df_mark) for i in range(CPU_CORES_COUNT)])
            # pool.map(run_indefinitely, range(CPU_CORES_COUNT))
        report_df = read_csv(REPORT_FULL_PATH)
        report_df.sort_values(report_df.columns[0], inplace=True)
        report_df.to_csv(REPORT_FULL_PATH, index=False)
        exec_time = time() - start_t
        eps = (CPU_CORES_COUNT * EPISODES_PER_CORE) / exec_time
        print(f'POOL exec time: {exec_time:.2f}s EpisodesPerSecond: {eps:.3f} StepsPerSecond: {eps * len(df):_.0f}')
        # sleep(10000)
        break


if __name__ == "__main__":
    main()
