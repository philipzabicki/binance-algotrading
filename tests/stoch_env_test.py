from multiprocessing import cpu_count, Pool
from statistics import mean, stdev

import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf

from definitions import ADDITIONAL_DATA_BY_MA
from enviroments import StochOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_StochasticOscillator, StochasticOscillator_signal

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 8_640
TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '5m', 'um', 'klines', '2020-01-01'
ENV = StochOptimizeSavingFuturesEnv
ACTION = [0.3465209168803213, 0.5951004853873286, 0.010648180318328126, 0.09175892106821656, 0.7488095334013507, 0.5686028928648609, 0.8690559570740761, 0.650230803068742, 45.416244520914944, 57.83957581237964, 243, 127, 628, 2, 9, 39]


def parallel_test(pool_nb, df, df_mark=None, dates_df=None):
    env = ENV(df=df,
              df_mark=df_mark,
              dates_df=dates_df,
              max_steps=N_STEPS,
              init_balance=50,
              no_action_finish=inf,
              fee=0.0005,
              coin_step=0.001,
              # slipp_std=0,
              # slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              verbose=False, visualize=False, write_to_file=True)
    results, gains = [], []
    for _ in range(N_TEST // CPU_CORES):
        _, reward, _, _, _ = env.step(ACTION)
        results.append(reward)
        if reward > 0:
            gains.append(env.exec_env.balance - env.exec_env.init_balance)
    return results, gains


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.25:
        return 0.25
    elif 0.25 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


if __name__ == "__main__":
    df = by_BinanceVision(ticker=TICKER,
                          interval=ITV,
                          market_type=MARKET_TYPE,
                          data_type=DATA_TYPE,
                          start_date=START_DATE,
                          split=False,
                          delay=345_600)
    _, df_mark = by_BinanceVision(ticker=TICKER,
                                  interval=ITV,
                                  market_type=MARKET_TYPE,
                                  data_type='markPriceKlines',
                                  start_date=START_DATE,
                                  split=True,
                                  delay=345_600)
    additional_periods = N_STEPS + ACTION[-6] + ACTION[-5] * ADDITIONAL_DATA_BY_MA[ACTION[-3]] + ACTION[-4] * \
                  ADDITIONAL_DATA_BY_MA[ACTION[-2]]
    slowK, slowD = custom_StochasticOscillator(df.iloc[-additional_periods:, 1:6].to_numpy(),
                                               fastK_period=ACTION[-6],
                                               slowK_period=ACTION[-5],
                                               slowD_period=ACTION[-4],
                                               slowK_ma_type=ACTION[-3],
                                               slowD_ma_type=ACTION[-2])
    signals = StochasticOscillator_signal(slowK,
                                          slowD,
                                          oversold_threshold=ACTION[-3],
                                          overbought_threshold=ACTION[-2])
    df_plot = df.tail(N_STEPS).copy()
    df_plot['slowK'] = slowK[-N_STEPS:]
    df_plot['slowD'] = slowD[-N_STEPS:]
    df_plot['signals'] = signals[-N_STEPS:]
    df_plot.index = pd.DatetimeIndex(df_plot['Opened'])

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    axs = gs.subplots(sharex=False)
    mpf.plot(df_plot, type='candle', style='yahoo', ylabel='Price', ax=axs[0])
    axs[1].plot(df_plot.index, df_plot['slowK'], label='slowK')
    axs[1].plot(df_plot.index, df_plot['slowD'], label='slowD')
    axs[1].axhline(y=ACTION[-8], label='oversold threshold', color='grey', linestyle='--')
    axs[1].axhline(y=ACTION[-7], label='overbought threshold', color='grey', linestyle='--')
    axs[1].axhline(y=50, color='black', linestyle='dashed')
    axs[1].legend(loc='upper left')
    axs[2].plot(df_plot.index, df_plot['signals'], label='Trade signals')
    axs[2].axhline(y=sig_map(ACTION[3]), label='Long enter threshold', color='green', linestyle='--')
    axs[2].axhline(y=-sig_map(ACTION[4]), label='Long close threshold', color='red', linestyle='--')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(parallel_test, [(i, df.iloc[:, 1:6], df_mark, df['Opened']) for i in range(CPU_CORES)])
    joined_res = []
    joined_gains = []
    for el in results:
        joined_res.extend(el[0])
        joined_gains.extend(el[1])
    profitable = sum(i > 0 for i in joined_res)
    print(f'From {len(joined_res)} tests, profitable: {profitable} ({profitable / len(joined_res) * 100}%)')
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${stdev(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
