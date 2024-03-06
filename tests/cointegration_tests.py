# Script used to find the best period to train models on based on finding 2 mostly cointegrated ones.
import pandas as pd
import numpy as np
from random import randint
from talib import *
from matplotlib import pyplot as plt
from utils.get_data import by_BinanceVision
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

TICKER = 'BTCUSDT'
ITV = '5m'
MARKET_TYPE = 'um'
DATA_TYPE = 'klines'
N_RND_SERIES = 100_000
N_LAST_INTERVALS = 20_160 # 80 days

if __name__ == "__main__":
    full_df = by_BinanceVision(ticker=TICKER,
                               interval=ITV,
                               market_type=MARKET_TYPE,
                               data_type=DATA_TYPE,
                               split=False,
                               delay=0)
    print(full_df)

    full_df['ADL'] = AD(full_df['High'], full_df['Low'], full_df['Close'], full_df['Volume'])
    full_df['OBV'] = OBV(full_df['Close'], full_df['Volume'])
    full_df['ADL'].ffill(inplace=True)
    full_df['OBV'].ffill(inplace=True)
    #full_df['AVGPRICE'] = AVGPRICE(full_df['Open'], full_df['High'], full_df['Low'], full_df['Close'])
    #full_df['AVGPRICE_standard'] = (full_df['AVGPRICE'] - np.mean(full_df['AVGPRICE'])) / np.std(full_df['AVGPRICE'])

    # full_df['MFM'] = ((full_df['Close'] - full_df['Low']) - (full_df['High'] - full_df['Close'])) / (
    #             full_df['High'] - full_df['Low'])
    # full_df['MFV'] = full_df['MFM'] * full_df['Volume']
    plt.plot(full_df['ADL'].to_numpy())
    plt.show()
    #full_df['ADL'].fillna(method='ffill', inplace=True)
    #full_df['OBV'].fillna(method='ffill', inplace=True)

    last_month = full_df.tail(N_LAST_INTERVALS)
    #last_month.loc[:, 'ADL'] = AD(last_month['High'], last_month['Low'], last_month['Close'], last_month['Volume'])

    for i in range(N_RND_SERIES):
        start_idx = randint(0, len(full_df) - 2 * N_LAST_INTERVALS)
        end_idx = start_idx + N_LAST_INTERVALS
        rnd_df = full_df.iloc[start_idx:end_idx, :]
        #rnd_df.loc[:, 'ADL'] = AD(rnd_df['High'], rnd_df['Low'], rnd_df['Close'], rnd_df['Volume'])

        # correlation_results = stats.pearsonr(rnd_df['MFV'], last_month['MFV'])
        # print(f'#{i} correlation {correlation_results}')
        # if abs(correlation_results[0]) >= 0.25 and correlation_results[1] <= 0.1:
        #     print(rnd_df['MFV'])
        #     print(last_month['MFV'])
        #     print(f'start_date: {rnd_df.iloc[0, 0]} end_date: {rnd_df.iloc[-1, 0]}')
        #     plt.plot(last_month['MFV'].to_numpy())
        #     plt.plot(rnd_df['MFV'].to_numpy())
        #     plt.show()

        results = coint(last_month['ADL'], rnd_df['ADL'])
        print(f'Test on random period #{i}: {results}')
        if results[1] < 0.1:
            print(rnd_df['ADL'])
            print(last_month['ADL'])
            print(f'start_date: {rnd_df.iloc[0, 0]} end_date: {rnd_df.iloc[-1, 0]}')
            scaler = MinMaxScaler()
            last_month.loc[:, 'ADL'] = scaler.fit_transform(last_month[['ADL']])
            rnd_df.loc[:, 'ADL'] = scaler.fit_transform(rnd_df[['ADL']])
            plt.plot(last_month['ADL'].to_numpy())
            plt.plot(rnd_df['ADL'].to_numpy())
            plt.show()
