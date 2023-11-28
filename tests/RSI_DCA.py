import TA_tools
from utility import minutes_since
from get_data import by_BinanceVision
import talib
import numpy as np


if __name__=='__main__':
    df = by_BinanceVision(ticker='BTCFDUSD',
                          interval='1m',
                          type='spot',
                          data='klines',
                          delay=1_000).tail(minutes_since('28-10-2023'))
    #df.drop(columns=['Opened'], inplace=True)
    print(f'df: {df}')

    rsi = talib.RSI(df['Close'], timeperiod=14)
    # print(f'rsi {rsi}')
    rsi_sig = np.array(TA_tools.RSI_like_signal(rsi.to_numpy(), 14))
    print(f'rsi_sig {rsi_sig}')

    # print(np.where(rsi_sig >= 1.0))
    count = np.sum(rsi_sig[rsi_sig == 1.0])
    print(f"Number of rows where all elements are equal to 1: {count}")