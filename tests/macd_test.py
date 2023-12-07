from get_data import by_BinanceVision
from utility import minutes_since
from matplotlib import pyplot as plt
from TA_tools import custom_MACD
from numpy import nan_to_num

if __name__ == '__main__':
    df = by_BinanceVision(ticker='BTCFDUSD',
                          interval='1m',
                          type='spot',
                          data='klines').tail(minutes_since('11-09-2023'))
    df = df.drop(columns='Opened').to_numpy()

    for i in range(35):
        for j in range(26):
            print(f'fast/slow: {i} signal {j}')
            macd, signal = custom_MACD(df, 12, 26, 9, i, i, j)
            # ma = nan_to_num(ma)
            # close = df['Close'].to_numpy()
            print(f'macd {macd}')
            print(f'signal {signal}')
            plt.plot(macd[-100:])
            plt.plot(signal[-100:])
            plt.show()