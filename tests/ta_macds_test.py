from matplotlib import pyplot as plt

from utils.get_data import by_BinanceVision
from utils.ta_tools import custom_MACD

if __name__ == '__main__':
    _, df = by_BinanceVision(ticker='BTCFDUSD',
                             interval='1m',
                             market_type='spot',
                             data_type='klines',
                             start_date='2023-09-11',
                             split=True,
                             delay=0)

    for i in range(35):
        for j in range(26):
            print(f'fast/slow: {i} signal {j}')
            macd, signal = custom_MACD(df.to_numpy(), 12, 26, 9, i, i, j)
            # ma = nan_to_num(ma)
            # close = df['Close'].to_numpy()
            print(f'macd {macd}')
            print(f'signal {signal}')
            plt.plot(macd[-100:])
            plt.plot(signal[-100:])
            plt.show()
