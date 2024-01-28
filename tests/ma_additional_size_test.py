import cProfile

from matplotlib import pyplot as plt
from numpy import round, array, mean, std
import matplotx

from utils.get_data import by_BinanceVision
from utils.ta_tools import get_1D_MA, get_MA


def check_ohlcv_mas(check_size, ma_period, precision):
    additional_data_ratios = []
    for t in range(27):
        # print(f'MAtype: {t}')
        additional_prev_data = 0
        result_full = round(get_1D_MA(df[:, 3], t, ma_period), precision)
        while True:
            ma = get_1D_MA(df[-ma_period - check_size - additional_prev_data:, 3], t, ma_period)
            # print(f'ma: {ma} type: {t} period: {ma_period}')
            result_short = round(ma, precision)
            try:
                assert list(result_full[-check_size:]) == list(result_short[-check_size:])
                # print(f'additional_prev_data: {additional_prev_data} ratio:{additional_prev_data / (ma_period+check_size):.2f}')
                # plt.plot(df[-check_size:, 3], label='close')
                # plt.plot(result_full[-check_size:], label='MA with additional data')
                # plt.plot(result_short[-check_size:], label='MA')
                # matplotx.line_labels()
                # plt.show()
                additional_data_ratios.append(additional_prev_data / ma_period)
                break
            except AssertionError:
                additional_prev_data += 5
    return additional_data_ratios


if __name__ == "__main__":
    '''This test allows to check how many data points (minimal) given MA type needs
    to calculate identical MA values as one calculating with full dataset'''
    dates, df = by_BinanceVision('BTCUSDT', '15m', 'um', 'klines', '2020-01-01', split=True)
    df = df.to_numpy()
    print(df)

    # profiler = cProfile.Profile()
    # profiler.enable()

    max_ma_period = 1000
    precision = 2
    print(f'max_ma_period: {max_ma_period} precision: {precision}')

    additional_ratios = []
    for period in range(10, max_ma_period, 10):
        check_size = period//8
        print(f'checking: ma_period: {period} check_size: {check_size}')
        ratios = check_ohlcv_mas(check_size, period, precision)
        additional_ratios.append(ratios)
        print(f'ratios per MA type {round(ratios, 0)}')
    additional_ratios = array(additional_ratios)
    #print(additional_ratios)

    ma_ratio = {}
    for i in range(additional_ratios.shape[1]):
        try:
            ma_ratio[i] = round(mean(additional_ratios[:, i]) + std(additional_ratios[:, i]), 0)
        except Exception as e:
            #print(e)
            ma_ratio[i] = 1
        if ma_ratio[i] < 1:
            ma_ratio[i] = 1
        plt.plot(additional_ratios[:, i], label='close')
        plt.show()
    print(ma_ratio)
    # profiler.disable()
    # profiler.print_stats(sort='tottime')
