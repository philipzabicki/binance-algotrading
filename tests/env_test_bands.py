from multiprocessing import Pool, cpu_count

from matplotlib import pyplot as plt
from numpy import inf, mean, std
import pandas as pd
import mplfinance as mpf
from talib import ATR

from definitions import ADDITIONAL_DATA_BY_OHLCV_MA
from enviroments.bands_env import BandsOptimizeSavingFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_MA_band_signal, get_MA

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 10_080
TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '1m', 'um', 'klines', '2020-01-01'
ENV = BandsOptimizeSavingFuturesEnv
ACTION = [0.019128504706745705, 0.16525064273566714, 0.011920990333916498, 0.3981783231172713, 0.6581033478867839, 0.7678786356708418, 0.550614570380608, 0.5907129495928004, 12.483203415141306, 945, 27, 817, 11]


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
    additional_periods = N_STEPS + max(ACTION[-2] * ADDITIONAL_DATA_BY_OHLCV_MA[ACTION[-3]],
                        ACTION[-4])
    ohlcv_np = df.iloc[:, 1:6].to_numpy()
    signals = get_MA_band_signal(ohlcv_np[-additional_periods:,], ACTION[-3], ACTION[-2], ACTION[-4], ACTION[-5])
    atr = ATR(ohlcv_np[-additional_periods:, 1], ohlcv_np[-additional_periods:, 2], ohlcv_np[-additional_periods:, 3], ACTION[-4])
    up_band = get_MA(ohlcv_np[-additional_periods:, ], ACTION[-3], ACTION[-2]) + atr*ACTION[-5]
    low_band = get_MA(ohlcv_np[-additional_periods:, ], ACTION[-3], ACTION[-2]) - atr*ACTION[-5]
    df_plot = df.tail(N_STEPS).copy()
    #df_plot['MACD'] = macd[-N_STEPS:]
    #df_plot['signal'] = signal[-N_STEPS:]
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
    axs[2].axhline(y=ACTION[4], label='Long open', color='green', linestyle='--')
    axs[2].axhline(y=-ACTION[5], label='Long close', color='red', linestyle='--')
    axs[2].axhline(y=-ACTION[6], label='Short open', color='red', linestyle='dotted')
    axs[2].axhline(y=ACTION[7], label='Short close', color='green', linestyle='dotted')
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
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${std(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
