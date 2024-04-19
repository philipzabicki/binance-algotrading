# Script used to find the best period to train models on based on finding 2 mostly cointegrated ones.
from random import randint

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import coint
from talib import *

from utils.get_data import by_BinanceVision

TICKER = 'BTCUSDT'
ITV = '5m'
MARKET_TYPE = 'um'
DATA_TYPE = 'klines'
N_RND_SERIES = 100_000
N_LAST_INTERVALS = 17_280  # 60 days

if __name__ == "__main__":
    full_df = by_BinanceVision(ticker=TICKER,
                               interval=ITV,
                               market_type=MARKET_TYPE,
                               data_type=DATA_TYPE,
                               split=False,
                               delay=0)
    print(full_df)

    full_df['ADL'] = AD(full_df['High'], full_df['Low'], full_df['Close'], full_df['Volume'])
    # full_df['RSI'] = RSI(full_df['Close'], timeperiod=14)
    # full_df['ULTOSC'] = ULTOSC(full_df['High'], full_df['Low'], full_df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # full_df['MFI'] = MFI(full_df['High'], full_df['Low'], full_df['Close'], full_df['Volume'], timeperiod=N_LAST_INTERVALS//100)
    full_df['ATR'] = ATR(full_df['High'], full_df['Low'], full_df['Close'], timeperiod=N_LAST_INTERVALS//100)
    full_df['OBV'] = OBV(full_df['Close'], full_df['Volume'])
    full_df['HL2'] = MEDPRICE(full_df['High'], full_df['Low'])
    # full_df['OBV'] = OBV(full_df['Close'], full_df['Volume'])
    full_df[['ADL', 'OBV', 'HL2', 'ATR']].ffill(inplace=True)
    # scaler_std = StandardScaler()
    # scaler_mm = MinMaxScaler()
    # full_df.loc[:, 'ADL'] = scaler_std.fit_transform(full_df[['ADL']])
    # full_df.loc[:, 'HL2'] = scaler_std.fit_transform(full_df[['HL2']])
    # full_df.loc[:, 'ATR'] = scaler_std.fit_transform(full_df[['ATR']])
    # full_df.loc[:, 'RSI'] = scaler_mm.fit_transform(full_df[['RSI']])
    # full_df.loc[:, 'ULTOSC'] = scaler_mm.fit_transform(full_df[['ULTOSC']])
    # full_df.loc[:, 'MFI'] = scaler_mm.fit_transform(full_df[['MFI']])
    full_df['target'] = full_df[['ADL', 'OBV', 'HL2', 'ATR']].sum(axis=1)
    # scaler = MinMaxScaler()
    # full_df.loc[:, 'HL2'] = scaler.fit_transform(full_df[['ADL']])
    # full_df['OBV'].ffill(inplace=True)
    # full_df['AVGPRICE'] = AVGPRICE(full_df['Open'], full_df['High'], full_df['Low'], full_df['Close'])
    # full_df['AVGPRICE_standard'] = (full_df['AVGPRICE'] - np.mean(full_df['AVGPRICE'])) / np.std(full_df['AVGPRICE'])

    # full_df['MFM'] = ((full_df['Close'] - full_df['Low']) - (full_df['High'] - full_df['Close'])) / (
    #             full_df['High'] - full_df['Low'])
    # full_df['MFV'] = full_df['MFM'] * full_df['Volume']
    count_0_ADL = (full_df['target'] == 0.0).sum()
    print(f"0.0 values in column 'target': {count_0_ADL}")
    plt.plot(full_df['target'].tail(N_LAST_INTERVALS).to_numpy())
    plt.show()
    # full_df['ADL'].fillna(method='ffill', inplace=True)
    # full_df['OBV'].fillna(method='ffill', inplace=True)

    last_month = full_df.tail(N_LAST_INTERVALS)
    # last_month.loc[:, 'ADL'] = AD(last_month['High'], last_month['Low'], last_month['Close'], last_month['Volume'])

    for i in range(N_RND_SERIES):
        start_idx = randint(0, len(full_df) - 2 * N_LAST_INTERVALS)
        end_idx = start_idx + N_LAST_INTERVALS
        rnd_df = full_df.iloc[start_idx:end_idx, :]
        # rnd_df.loc[:, 'ADL'] = AD(rnd_df['High'], rnd_df['Low'], rnd_df['Close'], rnd_df['Volume'])

        # correlation_results = stats.pearsonr(rnd_df['MFV'], last_month['MFV'])
        # print(f'#{i} correlation {correlation_results}')
        # if abs(correlation_results[0]) >= 0.25 and correlation_results[1] <= 0.1:
        #     print(rnd_df['MFV'])
        #     print(last_month['MFV'])
        #     print(f'start_date: {rnd_df.iloc[0, 0]} end_date: {rnd_df.iloc[-1, 0]}')
        #     plt.plot(last_month['MFV'].to_numpy())
        #     plt.plot(rnd_df['MFV'].to_numpy())
        #     plt.show()

        results = coint(last_month['target'], rnd_df['target'])
        print(f'Test on random period #{i}: {results}')
        if results[1] < 0.05:
            print(rnd_df['target'])
            print(last_month['target'])
            print(f'start_date: {rnd_df.iloc[0, 0]} end_date: {rnd_df.iloc[-1, 0]}')
            # last_month.loc[:, 'target'] = scaler1.fit_transform(last_month[['target']])
            # rnd_df.loc[:, 'target'] = scaler2.fit_transform(rnd_df[['target']])
            plt.plot(MinMaxScaler().fit_transform(last_month[['target']]))
            plt.plot(MinMaxScaler().fit_transform(rnd_df[['target']]))
            plt.show()
