import mplfinance as mpf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf
from talib import AD

from enviroments import MACDStratSpotEnv, MACDStratFuturesEnv
from enviroments.chaikinosc_env import ChaikinOscillatorStratSpotEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_ChaikinOscillator, get_1D_MA, ChaikinOscillator_signal
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
    action = [0.014984120484895906, 627, 152, 9, 25]

    # df = pd.read_csv("C:/github/binance-algotrading/.other/lotos.csv")
    df = by_BinanceVision(ticker=ticker,
                          interval=interval,
                          market_type=market_type,
                          data_type=data_type,
                          start_date=start_date,
                          split=False,
                          delay=259_200)
    # _, df_mark = by_BinanceVision(ticker='BTCUSDT',
    #                               interval='1m',
    #                               market_type='um',
    #                               data_type='markPriceKlines',
    #                               start_date='2021-01-01',
    #                               split=True,
    #                               delay=259_200)
    adl = AD(df['High'], df['Low'], df['Close'], df['Volume']).to_numpy()
    fast_adl, slow_adl = get_1D_MA(adl, action[3], action[1]), get_1D_MA(adl, action[4], action[2])
    chosc = fast_adl - slow_adl
    signals = ChaikinOscillator_signal(chosc)
    df['fast_ADL'] = fast_adl
    df['slow_ADL'] = slow_adl
    df['ChaikinOscillator'] = chosc
    df['signals'] = signals
    # df = df.tail(250)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    axs = gs.subplots(sharex=False)
    axs[0].plot(df['fast_ADL'], label='fast_ADL')
    axs[0].plot(df['slow_ADL'], label='slow_ADL')
    axs[0].legend(loc='upper left')
    axs[1].plot(df['ChaikinOscillator'], label='ChaikinOscillator')
    axs[1].legend(loc='upper left')
    axs[2].plot(df['signals'], label='Trade signals')
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

    env = ChaikinOscillatorStratSpotEnv(df=df.iloc[:, 1:6],
                                        dates_df=df['Opened'],
                                        # max_steps=259_200,
                                        init_balance=350,
                                        no_action_finish=inf,
                                        fee=0.0,
                                        coin_step=0.00001,
                                        # slipp_std=0,
                                        # slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                                        verbose=True, visualize=False, write_to_file=True)
    env.step(action)
