from multiprocessing import cpu_count, Pool
from statistics import mean, stdev, median

import mplfinance as mpf
import pandas as pd
from numpy import quantile
from matplotlib import pyplot as plt
from numpy import inf

from definitions import ADDITIONAL_DATA_BY_MA, ADDITIONAL_DATA_BY_OHLCV_MA
from enviroments import MACDOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_MACD, MACD_cross_signal

CPU_CORES = cpu_count()
# CPU_CORES = 1
N_TEST = 10_000
N_STEPS = 8_640
TICKER, ITV, MARKET_TYPE, DATA_TYPE = 'BTCUSDT', '5m', 'um', 'klines'
TRADE_START_DATE = '2021-11-20'
TRADE_END_DATE = '2022-03-20'
# Better to take more previous data for some TA features
DF_START_DATE = '2021-07-20'
DF_END_DATE = '2022-03-21'
ENV = MACDOptimizeSavingFuturesEnv
ACTION = [0.7068344840046556, 0.06866980724047726, 0.014667508215968673, 0.6624315457484368, 0.9492528324000956, 0.05215074097490277, 0.910665238610637, 0.6015669284499915, 12, 428, 618, 19, 33, 23, 43]


def parallel_test(pool_nb, df, df_mark=None):
    env = ENV(df=df,
              df_mark=df_mark,
              start_date=TRADE_START_DATE,
              end_date=TRADE_END_DATE,
              max_steps=N_STEPS,
              init_balance=50,
              no_action_finish=inf,
              fee=0.0005,
              coin_step=0.001,
              # slipp_std=0,
              # slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              verbose=False, visualize=False, write_to_file=False)
    results, gains = [], []
    for _ in range(N_TEST // CPU_CORES):
        _, reward, _, _, _ = env.step(ACTION)
        results.append(reward)
        _diff = env.exec_env.balance - env.exec_env.init_balance
        if reward > 0 and _diff > 0:
            gains.append(_diff)
    return results, gains


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
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
    if N_STEPS > 0:
        _size = N_STEPS
        additional_periods = _size + 1 * (max(ACTION[-7] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[-4]],
                                              ACTION[-6] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[-3]]) + ACTION[-5] *
                                          ADDITIONAL_DATA_BY_MA[ACTION[-2]])
        print(f'additional_periods={additional_periods}')
        macd, signal = custom_MACD(df.iloc[-additional_periods:, 1:6].to_numpy(), ACTION[-7], ACTION[-6], ACTION[-5],
                                   ACTION[-4], ACTION[-3],
                                   ACTION[-2])
        signals = MACD_cross_signal(macd, signal)
        df_plot = df.tail(_size).copy()
        df_plot['MACD'] = macd[-_size:]
        df_plot['signal'] = signal[-_size:]
        df_plot['signals'] = signals[-_size:]
        df_plot.index = pd.DatetimeIndex(df_plot['Opened'])
    else:
        additional_periods = max(ACTION[-7] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[-4]],
                                 ACTION[-6] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[-3]]) + ACTION[-5] * \
                             ADDITIONAL_DATA_BY_MA[ACTION[-2]]
        print(f'additional_periods={additional_periods}')
        macd, signal = custom_MACD(df.iloc[:, 1:6].to_numpy(), ACTION[-7], ACTION[-6], ACTION[-5],
                                   ACTION[-4], ACTION[-3],
                                   ACTION[-2])
        signals = MACD_cross_signal(macd, signal)
        df_plot = df.tail(len(df) - additional_periods).copy()
        df_plot['MACD'] = macd[additional_periods:]
        df_plot['signal'] = signal[additional_periods:]
        df_plot['signals'] = signals[additional_periods:]
        df_plot.index = pd.DatetimeIndex(df_plot['Opened'])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    axs = gs.subplots(sharex=False)
    mpf.plot(df_plot, type='candle', style='yahoo', ylabel='Price', ax=axs[0])
    axs[1].plot(df_plot.index, df_plot['MACD'], label='MACD')
    axs[1].plot(df_plot.index, df_plot['signal'], label='Signal')
    axs[1].legend(loc='upper left')
    axs[2].plot(df_plot.index, df_plot['signals'], label='Trade signals')
    axs[2].axhline(y=sig_map(ACTION[3]), label='Buy threshold', color='green', linestyle='--')
    axs[2].axhline(y=-sig_map(ACTION[4]), label='Sell threshold', color='red', linestyle='--')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(parallel_test, [(i, df, df_mark) for i in range(CPU_CORES)])
    joined_res = []
    joined_gains = []
    for el in results:
        joined_res.extend(el[0])
        joined_gains.extend(el[1])
    profitable = sum(i > 0 for i in joined_res)
    print(f'From {len(joined_res)} tests, profitable: {profitable} ({profitable / len(joined_res) * 100}%)')
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${stdev(joined_gains):_.2f}')
    print(f'gain(quartiles) ${quantile(joined_gains, [0.25,0.5,0.75,1])}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
