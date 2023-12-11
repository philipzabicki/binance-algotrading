import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf

from enviroments import MACDStratSpotEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_MACD, MACD_cross_signal
from utils.utility import get_slippage_stats


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


if __name__ == "__main__":
    ticker, interval, market_type, data_type, start_date = 'BTCFDUSD', '1m', 'spot', 'klines', '2023-09-11'
    action = [0.011123635861121386, 0.32212581146447883, 0.5566056106449668, 586, 810, 970, 9, 4, 23]

    df = by_BinanceVision(ticker=ticker,
                          interval=interval,
                          market_type=market_type,
                          data_type=data_type,
                          start_date=start_date,
                          split=False,
                          delay=129_600)
    macd, signal = custom_MACD(df.iloc[:, 1:6].to_numpy(), action[3], action[4], action[5], action[6], action[7],
                               action[8])
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
    axs[2].axhline(y=sig_map(action[1]), label='Buy threshold', color='green', linestyle='--')
    axs[2].axhline(y=-sig_map(action[2]), label='Sell threshold', color='red', linestyle='--')
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

    env = MACDStratSpotEnv(df=df.iloc[:, 1:6], dates_df=df['Opened'], init_balance=300, no_action_finish=inf,
                           fee=0.0, coin_step=0.00001,
                           slippage=get_slippage_stats(market_type, ticker, interval, 'market'),
                           verbose=True, visualize=False)
    env.step(action)
