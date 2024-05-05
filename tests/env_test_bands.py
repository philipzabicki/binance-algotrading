from multiprocessing import Pool, cpu_count

import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf, mean, std
from talib import ATR

from definitions import ADDITIONAL_DATA_BY_OHLCV_MA
from enviroments.bands_env import BandsOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_MA_band_signal, get_MA

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 8_640
TICKER, ITV, MARKET_TYPE, DATA_TYPE = 'BTCUSDT', '5m', 'um', 'klines'
TRADE_START_DATE = '2021-04-07'
TRADE_END_DATE = '2021-08-05'
# Better to take more previous data for some TA features
DF_START_DATE = '2021-01-07'
DF_END_DATE = '2021-08-06'
ENV = BandsOptimizeSavingFuturesEnv
ACTION = [15, 21, 244, 22, 242, 3, 0.007323772650153117, 0.9231644815460224, 0.03336436648676175, 0.036715294645426, 0.24368303659766918, 0.937191485799508, 10.455677675503289]


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
              verbose=False, visualize=False, write_to_file=True)
    results, gains = [], []
    for _ in range(N_TEST // CPU_CORES):
        _, reward, _, _, _ = env.step(ACTION)
        results.append(reward)
        _diff = env.exec_env.balance - env.exec_env.init_balance
        if reward > 0 and _diff > 0:
            gains.append(_diff)
    return results, gains


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
    additional_periods = N_STEPS + max(ACTION[4] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[3]],
                                       ACTION[2])
    print(f'ACTION {ACTION}')
    ohlcv_np = df.iloc[:, 1:6].to_numpy()
    signals = get_MA_band_signal(ohlcv_np[-additional_periods:, ], ACTION[3], ACTION[4], ACTION[2], ACTION[-1])
    atr = ATR(ohlcv_np[-additional_periods:, 1], ohlcv_np[-additional_periods:, 2], ohlcv_np[-additional_periods:, 3],
              ACTION[2])
    up_band = get_MA(ohlcv_np[-additional_periods:, ], ACTION[3], ACTION[4]) + atr * ACTION[12]
    low_band = get_MA(ohlcv_np[-additional_periods:, ], ACTION[3], ACTION[4]) - atr * ACTION[12]
    df_plot = df.tail(N_STEPS).copy()
    # df_plot['MACD'] = macd[-N_STEPS:]
    # df_plot['signal'] = signal[-N_STEPS:]
    df_plot['signals'] = signals[-N_STEPS:]
    df_plot['upper'] = up_band[-N_STEPS:]
    df_plot['lower'] = low_band[-N_STEPS:]
    df_plot.index = pd.DatetimeIndex(df_plot['Opened'])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    axs = gs.subplots(sharex=False)
    mpf.plot(df_plot, type='candle', style='yahoo', ylabel='Price', ax=axs[0])
    axs[1].plot(df_plot.index, df_plot['Close'], label='Close price')
    axs[1].plot(df_plot.index, df_plot['upper'], label='Upper band')
    axs[1].plot(df_plot.index, df_plot['lower'], label='Lower band')
    axs[1].legend(loc='upper left')
    axs[2].plot(df_plot.index, df_plot['signals'], label='Signal line')
    axs[2].axhline(y=ACTION[8], label='Long open', color='green', linestyle='--')
    axs[2].axhline(y=-ACTION[9], label='Long close', color='red', linestyle='--')
    axs[2].axhline(y=-ACTION[10], label='Short open', color='red', linestyle='dotted')
    axs[2].axhline(y=ACTION[11], label='Short close', color='green', linestyle='dotted')
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
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${std(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
