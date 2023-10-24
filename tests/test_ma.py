from get_data import by_BinanceVision
from TA_tools import get_MA
from matplotlib import pyplot as plt
import talib
from random import randint

if __name__=="__main__":
    df = by_BinanceVision('BTCFDUSD', '1s', 'spot')
    df = df.drop(columns=['Opened']).to_numpy()[-500_000:, :]
    print(df)
    # multiplayer = 1000
    #period = randint(2,1_000)
    period = 10
    print(f'period: {period}')
    check_size = 10

    for t in range(35):
        print(f'MAtype: {t}')
        length = 2

        additional_prev_data = 10
        # print(result_full[-check_size:])
        # print(result_short[-check_size:])
        result_full = get_MA(df, t, period)
        while True:
            result_short = get_MA(df[-period - additional_prev_data:, ], t, period)
            try:
                assert list(result_full[-check_size:]) == list(result_short[-check_size:])
                print(f'additional_prev_data: {additional_prev_data}')
                break
            except AssertionError:
                # print(f'i: {i} period: {period}')
                additional_prev_data += 10
                # period = randint(2, 1_000)
                result_short = get_MA(df[-period - additional_prev_data:, ], t, period)