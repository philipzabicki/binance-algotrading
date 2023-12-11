import numpy as np
import talib

from utils.get_data import by_BinanceVision
from utils.ta_tools import RSI_oversold_signal

if __name__ == '__main__':
    df = by_BinanceVision(ticker='BTCFDUSD',
                          interval='1m',
                          market_type='spot',
                          data_type='klines',
                          start_date='2023-09-11',
                          split=False,
                          delay=0)
    # df.drop(columns=['Opened'], inplace=True)
    print(f'df: {df}')

    rsi = talib.RSI(df['Close'], timeperiod=14)
    # print(f'rsi {rsi}')
    rsi_sig = np.array(RSI_oversold_signal(rsi.to_numpy(), 14, oversold_threshold=17.5))
    print(f'rsi_sig {rsi_sig}')

    # print(np.where(rsi_sig >= 1.0))
    count = np.sum(rsi_sig[rsi_sig == 1.0])
    print(f"Number of rows with RSI14 oversold_threshold=17.5 signal to buy: {count}")
