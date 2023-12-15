from random import randint
from time import time

from numpy import mean
from talib import LINEARREG

from utils.get_data import by_BinanceVision
from utils.ta_tools import LSMA


def test_speed(func, close, runs=10):
    print(f'testing {func.__name__}')
    times = []
    for _ in range(runs):
        start_t = time()
        _ = func(close, randint(2, 1_000))
        times.append(time() - start_t)
    return mean(times)


if __name__ == "__main__":
    _, df = by_BinanceVision(ticker='BTCFDUSD',
                             interval='1m',
                             market_type='spot',
                             data_type='klines',
                             start_date='2023-09-11',
                             split=True,
                             delay=0)
    df = df.to_numpy()

    print(f'{test_speed(LINEARREG, df[:, 3], 1000)/1_000}ms')
    print(f'{test_speed(LSMA, df[:, 3], 1000)/1_000}ms')

    print(f'LSMA-37: {LSMA(df[:, 3], 37)}')
    print(f'LINEARREG-37: {LINEARREG(df[:, 3], 37)}')
