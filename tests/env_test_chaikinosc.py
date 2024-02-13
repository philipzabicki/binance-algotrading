from multiprocessing import Pool, cpu_count
from statistics import mean, stdev

import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf
from talib import AD

from definitions import ADDITIONAL_DATA_BY_MA
from enviroments.chaikinosc_env import ChaikinOscillatorOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_1D_MA, ChaikinOscillator_signal

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 288
TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '5m', 'um', 'klines', '2020-01-01'
ENV = ChaikinOscillatorOptimizeSavingFuturesEnv
ACTION = [0.1477896216992889, 0.5894053393009026, 0.008756753034056002, 0.43414056462836986, 78, 85, 15, 15, 24]


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


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
    # print(f'pool{pool_nb}')
    # print(f'results{results}')
    # print(f'gains{gains}')
    return results, gains


if __name__ == "__main__":
    # df = pd.read_csv("C:/github/binance-algotrading/.other/lotos.csv")
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
    additional_periods = N_STEPS + max(ACTION[-5] * ADDITIONAL_DATA_BY_MA[ACTION[-3]],
                                       ACTION[-4] * ADDITIONAL_DATA_BY_MA[ACTION[-2]])
    adl = AD(df['High'], df['Low'], df['Close'], df['Volume']).to_numpy()
    # print(adl[:10])
    # sleep(100)
    fast_adl, slow_adl = get_1D_MA(adl[-additional_periods:], ACTION[-3], ACTION[-5]), get_1D_MA(
        adl[-additional_periods:], ACTION[-2], ACTION[-4])
    chosc = fast_adl - slow_adl
    signals = ChaikinOscillator_signal(chosc)
    df_plot = df.tail(N_STEPS).copy()
    df_plot['ADL'] = adl[-N_STEPS:]
    df_plot['fast_ADL'] = fast_adl[-N_STEPS:]
    df_plot['slow_ADL'] = slow_adl[-N_STEPS:]
    df_plot['ChaikinOscillator'] = chosc[-N_STEPS:]
    df_plot['signals'] = signals[-N_STEPS:]
    df_plot.index = pd.DatetimeIndex(df_plot['Opened'])
    # df = df.tail(250)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
    axs = gs.subplots(sharex=False)
    mpf.plot(df_plot, type='candle', style='yahoo', ylabel='Price', ax=axs[0])
    axs[1].plot(df_plot['ADL'], label='ADL', linestyle='solid')
    axs[1].plot(df_plot['fast_ADL'], label='fast_ADL', linestyle=(0, (5, 1)))
    axs[1].plot(df_plot['slow_ADL'], label='slow_ADL', linestyle=(0, (5, 2)))
    axs[1].legend(loc='upper left')
    axs[2].plot(df_plot['ChaikinOscillator'], label='ChaikinOscillator')
    axs[2].axhline(y=0, label='Zero-line', color='black', linestyle='dashed')
    axs[2].legend(loc='upper left')
    axs[3].plot(df_plot['signals'], label='Trade signals')
    axs[3].axhline(y=1, label='Buy threshold', color='green', linestyle='dotted')
    axs[3].axhline(y=-1, label='Sell threshold', color='red', linestyle='dotted')
    axs[3].legend(loc='upper left')

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
