from pandas import read_csv
import datetime as dt
from dateutil.parser import parse
# from scipy.stats import skew, kurtosis
from pympler import asizeof
from os import path
from definitions import SLIPPAGE_DIR


def get_slippage_stats(market_type, ticker, interval, order_type='market'):
    buy_file = f'{SLIPPAGE_DIR}{market_type}/{ticker}{interval}/{order_type}_buy.csv'
    sell_file = f'{SLIPPAGE_DIR}{market_type}/{ticker}{interval}/{order_type}_sell.csv'
    stop_loss_file = f'{SLIPPAGE_DIR}{market_type}/{ticker}{interval}/limit_stop_loss.csv'
    if path.isfile(buy_file) and path.isfile(sell_file) and path.isfile(stop_loss_file):
        buy = read_csv(buy_file)
        sell = read_csv(sell_file)
        stop_loss = read_csv(stop_loss_file)
        slippages = {'buy': (buy.values.mean(), buy.values.std()),
                     'sell': (sell.values.mean(), sell.values.std()),
                     'stop_loss': (stop_loss.values.mean(), stop_loss.values.std())}
    else:
        slippages = {'buy': (1.0, 0.0),
                     'sell': (1.0, 0.0),
                     'stop_loss': (1.0, 0.0)}
        print(f'{ticker} {interval} {market_type} {order_type} SLIPPAGES DATA DOES NOT EXISTS,', end=' ')
        print(f'USING {slippages} INSTEAD')
    # print(slipps)
    return slippages



'''def get_stats_for_file(file_path):
    df = read_csv(file_path, header=0)
    mean = float(df.mean().values[0])
    std = float(df.std().values[0])
    return mean, std

def get_slips_stats():
    base_path = ROOT_DIR + '\\settings\\'
    file_names = ['slippages_market_buy.old2.csv', 'slippages_market_sell.old2.csv', 'limit_stop_loss.csv']
    labels = ['market_buy', 'market_sell', 'SL']
    stats = { label:get_stats_for_file(base_path + file_name) for label,file_name in zip(labels,file_names) }
    return stats'''

'''def get_slips_stats():
    buy = read_csv(ROOT_DIR+'/settings/slippages_market_buy.old2.csv', header=0)
    sell = read_csv(ROOT_DIR+'/settings/slippages_market_sell.old2.csv', header=0)
    SL = read_csv(ROOT_DIR+'/settings/limit_stop_loss.csv', header=0)
    return {'market_buy':(float(np.mean(buy)), float(np.std(buy))),
            'market_sell':(float(np.mean(sell)), float(np.std(sell))),
            'SL':(float(np.mean(SL)), float(np.std(SL)))}'''


# def get_slips_stats_advanced():
#     buy = read_csv(ROOT_DIR + '/settings/slippages_market_buy.old2.csv')
#     sell = read_csv(ROOT_DIR + '/settings/slippages_market_buy.old2.csv')
#     SL = read_csv(ROOT_DIR + '/settings/slippages_market_buy.old2.csv')
#     return {
#         'market_buy': {
#             'mean': buy.mean(),
#             'std': buy.std(),
#             'skewness': buy.apply(skew),
#             'kurtosis': buy.apply(kurtosis)
#         },
#         'market_sell': {
#             'mean': sell.mean(),
#             'std': sell.std(),
#             'skewness': sell.apply(skew),
#             'kurtosis': sell.apply(kurtosis)
#         },
#         'SL': {
#             'mean': SL.mean(),
#             'std': SL.std(),
#             'skewness': SL.apply(skew),
#             'kurtosis': SL.apply(kurtosis)
#         }
#     }


def minutes_since(data_string):
    diff = dt.datetime.now() - parse(data_string, dayfirst=True)
    minutes = diff.total_seconds() / 60
    return int(minutes)


def seconds_since(data_string):
    diff = dt.datetime.now() - parse(data_string, dayfirst=True)
    seconds = diff.total_seconds()
    return int(seconds)


def get_attributes_and_deep_sizes(obj):
    attributes_and_sizes = {}
    for attribute_name in dir(obj):
        attribute_value = getattr(obj, attribute_name)
        _size = asizeof.asizeof(attribute_value)
        if _size > 1_000:
            attributes_and_sizes[attribute_name] = asizeof.asizeof(attribute_value)
    return attributes_and_sizes
