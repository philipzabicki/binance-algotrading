# Script used to find the best period to train models on based on finding 2 mostly cointegrated ones.
import pandas as pd
import numpy as np
from random import randint
from talib import AD, AVGPRICE
from matplotlib import pyplot as plt
from utils.get_data import by_BinanceVision
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

TICKER = 'BTCUSDT'
ITV = '5m'
MARKET_TYPE = 'um'
DATA_TYPE = 'klines'
N_RND_SERIES = 1_000
N_INTERVALS = 24_192

if __name__ == "__main__":
    full_df = by_BinanceVision(ticker=TICKER,
                               interval=ITV,
                               market_type=MARKET_TYPE,
                               data_type=DATA_TYPE,
                               split=False,
                               delay=0)
    print(full_df)

    full_df['ADL'] = AD(full_df['High'], full_df['Low'], full_df['Close'], full_df['Volume'])
    #full_df['AVGPRICE'] = AVGPRICE(full_df['Open'], full_df['High'], full_df['Low'], full_df['Close'])
    #full_df['AVGPRICE_standard'] = (full_df['AVGPRICE'] - np.mean(full_df['AVGPRICE'])) / np.std(full_df['AVGPRICE'])
    # # Calculate Money Flow Multiplier
    # full_df['MFM'] = ((full_df['Close'] - full_df['Low']) - (full_df['High'] - full_df['Close'])) / (
    #             full_df['High'] - full_df['Low'])
    # # Handle potential division by zero if High equals Low by replacing infinities and NaNs with 0 (or an appropriate value)
    # full_df['MFM'].replace([float('inf'), -float('inf'), pd.NA], 0, inplace=True)
    # # Calculate Money Flow Volume
    # full_df['MFV'] = full_df['MFM'] * full_df['Volume']

    # Display the first few rows to confirm calculations
    # full_df['MFV_standard'] = (full_df['MFV'] - np.mean(full_df['MFV'])) / np.std(full_df['MFV'])

    last_month = full_df.tail(N_INTERVALS)

    for i in range(N_RND_SERIES):
        start_idx = randint(0, len(full_df) - 2 * N_INTERVALS)
        end_idx = start_idx + N_INTERVALS
        rnd_df = full_df.iloc[start_idx:end_idx, :]

        results = coint(last_month['ADL'], rnd_df['ADL'])
        print(f'Test on random period #{i}: {results}')
        if results[1] < 0.05:
            print(rnd_df)
            print(last_month['ADL'])
            print(f'start_date: {rnd_df.iloc[0, 0]} end_date: {rnd_df.iloc[-1, 0]}')
            plt.plot(last_month['ADL'].to_numpy())
            plt.plot(rnd_df['ADL'].to_numpy())
            plt.show()
