from statistics import mean, stdev

import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf

from enviroments import MACDStratFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_MACD, MACD_cross_signal
from utils.utility import get_slippage_stats

N_TEST = 1000


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


if __name__ == "__main__":
    ticker, interval, market_type, data_type, start_date = 'BTCUSDT', '15m', 'um', 'klines', '2020-01-01'
    action = [0.9839826807196196, 0.01026489549333572, 0.02022990188305404, 0.5164379927641636, 531, 644, 240, 10, 31, 21, 65]

    df = by_BinanceVision(ticker=ticker,
                          interval=interval,
                          market_type=market_type,
                          data_type=data_type,
                          start_date=start_date,
                          split=False,
                          delay=259_200)
    _, df_mark = by_BinanceVision(ticker=ticker,
                                  interval=interval,
                                  market_type=market_type,
                                  data_type='markPriceKlines',
                                  start_date=start_date,
                                  split=True,
                                  delay=259_200)
    macd, signal = custom_MACD(df.iloc[:, 1:6].to_numpy(), action[-7], action[-6], action[-5], action[-4], action[-3],
                               action[-2])
    signals = MACD_cross_signal(macd, signal)
    df['MACD'] = macd
    df['signal'] = signal
    df['signals'] = signals
    df.index = pd.DatetimeIndex(df['Opened'])
    df_plot = df.tail(500)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    axs = gs.subplots(sharex=False)
    mpf.plot(df_plot, type='candle', style='yahoo', ylabel='Price', ax=axs[0])
    axs[1].plot(df_plot.index, df_plot['MACD'], label='MACD')
    axs[1].plot(df_plot.index, df_plot['signal'], label='Signal')
    axs[1].legend(loc='upper left')
    axs[2].plot(df_plot.index, df_plot['signals'], label='Trade signals')
    axs[2].axhline(y=sig_map(action[2]), label='Buy threshold', color='green', linestyle='--')
    axs[2].axhline(y=-sig_map(action[3]), label='Sell threshold', color='red', linestyle='--')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # plt.subplot(2, 1, 1)
    # plt.plot(macd[-1_000:], label='MACD')
    # plt.plot(signal[-1_000:], label='Signal')
    # plt.legend()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(signals[-1_000:], label='Signals')
    # plt.axhline(y=sig_map(action[1]), color='green', linestyle='--')
    # plt.axhline(y=-sig_map(action[2]), color='red', linestyle='--')
    # plt.legend()
    #
    # plt.show()

    env = MACDStratFuturesEnv(df=df.iloc[:, 1:6],
                              df_mark=df_mark,
                              dates_df=df['Opened'],
                              max_steps=2_880,
                              init_balance=350,
                              no_action_finish=inf,
                              fee=0.0005,
                              coin_step=0.001,
                              slipp_std=0,
                              slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                              verbose=False, visualize=False, write_to_file=True)
    results = []
    gains = []
    for _ in range(N_TEST):
        _, reward, _, _, _ = env.step(action)
        results.append(reward)
        if reward > 0:
            gains.append(env.exec_env.balance - env.exec_env.init_balance)
    profitable = sum(i > 0 for i in results)
    print(f'From {N_TEST} tests, profitable: {profitable} ({profitable / len(results) * 100}%)')
    print(f'gain(avg/stdev): ${mean(gains):_.2f}/${stdev(gains):_.2f}')
    print(f'gain(min/max): ${min(gains):_.2f}/${max(gains):_.2f}')
