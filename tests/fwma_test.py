from random import randint
from time import time

from utils.get_data import by_BinanceVision
from utils.ta_tools import FWMA, CWMA


if __name__ == "__main__":
    dates, df = by_BinanceVision(ticker='BTCFDUSD',
                             interval='1m',
                             market_type='spot',
                             data_type='klines',
                             start_date='2023-09-11',
                             split=True,
                             delay=0)
    df = df.to_numpy()
    print(f'dates {dates}')

    print(FWMA(df[:, 3], 10))