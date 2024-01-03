import cProfile

from matplotlib import pyplot as plt
from numpy import around

from utils.get_data import by_BinanceVision
from utils.ta_tools import get_MA

if __name__ == "__main__":
    '''This test allows to check how many data points (minimal) given MA type needs
    to calculate identical MA values as one calculating with full dataset'''
    df = by_BinanceVision('BTCFDUSD', '1m', 'spot')
    df = df.drop(columns=['Opened']).to_numpy()[-50_000:, :]
    print(df)

    ma_period = 25
    print(f'ma_period: {ma_period}')
    check_size = 100
    precision = 4

    profiler = cProfile.Profile()
    profiler.enable()

    for t in range(37):
        print(f'MAtype: {t}')
        additional_prev_data = 0
        # print(result_full[-check_size:])
        # print(result_short[-check_size:])
        result_full = around(get_MA(df, t, ma_period), precision)
        # print(result_full)
        while True:
            ma = get_MA(df[-ma_period - additional_prev_data:, ], t, ma_period)
            result_short = around(ma, precision)
            try:
                assert list(result_full[-check_size:]) == list(result_short[-check_size:])
                print(f'additional_prev_data: {additional_prev_data} ratio:{additional_prev_data / ma_period:.2f}')
                # print(f'result_full: {result_full[-check_size:]}')
                # print(f'result_short: {result_short[-check_size:]}')
                plt.plot(df[-check_size:, 3])
                plt.plot(result_full[-check_size:])
                plt.plot(result_short[-check_size:])
                plt.show()
                break
            except AssertionError:
                # print(f'i: {i} period: {period}')
                additional_prev_data += 10
                # period = randint(2, 1_000)

    profiler.disable()
    profiler.print_stats(sort='tottime')
