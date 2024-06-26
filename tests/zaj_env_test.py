from multiprocessing import Pool, cpu_count
from statistics import mean, stdev

import pandas as pd
from enviroments.zajeciowy_env import ChaikinOscillatorStratSpotEnv
from matplotlib import pyplot as plt
from numpy import inf
from talib import AD

from utils.ta_tools import get_1D_MA, ChaikinOscillator_signal

CPU_CORES = cpu_count()
CPU_CORES = 1
N_TEST = 1
# N_STEPS = 5_760
# TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '15m', 'um', 'klines', '2020-01-01'
ENV = ChaikinOscillatorStratSpotEnv
ACTION = [20, 5, 6, 5]


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


def parallel_test(pool_nb, df, dates_df=None, df_mark=None):
    env = ENV(df=df,
              # df_mark=df_mark,
              dates_df=dates_df,
              # max_steps=N_STEPS,
              init_balance=1_000,
              no_action_finish=inf,
              fee=0.0,
              coin_step=0.01,
              # slipp_std=0,
              # slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              render_range=180,
              verbose=True, visualize=False, write_to_file=True)
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
    df = pd.read_csv("C:/github/binance-algotrading/.other/lotos.csv")
    df['Opened'] = pd.to_datetime(df['Opened'], format='%Y%m%d')
    print(df)

    adl = AD(df['High'], df['Low'], df['Close'], df['Volume']).to_numpy()
    # print(adl[:10])
    # sleep(100)
    fast_adl, slow_adl = get_1D_MA(adl, ACTION[2], ACTION[0]), get_1D_MA(adl, ACTION[3], ACTION[1])
    chosc = fast_adl - slow_adl
    signals = ChaikinOscillator_signal(chosc)
    df['ADL'] = adl
    df['fast_ADL'] = fast_adl
    df['slow_ADL'] = slow_adl
    df['ChaikinOscillator'] = chosc
    df['signals'] = signals
    # df = df.tail(250)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    axs = gs.subplots(sharex=False)
    axs[0].plot(df['ADL'], label='ADL', linestyle='solid')
    axs[0].plot(df['fast_ADL'], label='fast_ADL', linestyle=(0, (5, 1)))
    axs[0].plot(df['slow_ADL'], label='slow_ADL', linestyle=(0, (5, 2)))
    axs[0].legend(loc='upper left')
    axs[1].plot(df['ChaikinOscillator'], label='ChaikinOscillator')
    axs[1].axhline(y=0, label='Zero-line', color='black', linestyle='dashed')
    axs[1].legend(loc='upper left')
    axs[2].plot(df['signals'], label='Trade signals')
    axs[2].axhline(y=1, label='Buy threshold', color='green', linestyle='dotted')
    axs[2].axhline(y=-1, label='Sell threshold', color='red', linestyle='dotted')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(parallel_test, [(i, df.iloc[:, 1:6], df['Opened']) for i in range(CPU_CORES)])
    joined_res = []
    joined_gains = []
    for el in results:
        joined_res.extend(el[0])
        joined_gains.extend(el[1])
    profitable = sum(i > 0 for i in joined_res)
    print(f'From {len(joined_res)} tests, profitable: {profitable} ({profitable / len(joined_res) * 100}%)')
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${stdev(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
