from pandas import read_csv
import datetime as dt
from dateutil.parser import parse
from scipy.stats import skew, kurtosis
from pympler import asizeof
from definitions import ROOT_DIR

def get_market_slips_stats():
    buy = read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    sell = read_csv(ROOT_DIR + '/settings/slippages_market_sell.csv')
    SL = read_csv(ROOT_DIR + '/settings/slippages_StopLoss.csv')
    slipps = {'market_buy': (buy.values.mean(), buy.values.std()),
              'market_sell': (sell.values.mean(), sell.values.std()), 'SL': (SL.values.mean(), SL.values.std())}
    # print(slipps)
    return slipps


def get_limit_slips_stats():
    buy = read_csv(ROOT_DIR + '/settings/slippages_limit_buy.csv')
    sell = read_csv(ROOT_DIR + '/settings/slippages_limit_sell.csv')
    SL = read_csv(ROOT_DIR + '/settings/slippages_StopLoss.csv')
    slipps = {'market_buy': (buy.values.mean(), buy.values.std()),
              'market_sell': (sell.values.mean(), sell.values.std()), 'SL': (SL.values.mean(), SL.values.std())}
    # print(slipps)
    return slipps


'''def get_stats_for_file(file_path):
    df = read_csv(file_path, header=0)
    mean = float(df.mean().values[0])
    std = float(df.std().values[0])
    return mean, std

def get_slips_stats():
    base_path = ROOT_DIR + '\\settings\\'
    file_names = ['slippages_market_buy.csv', 'slippages_market_sell.csv', 'slippages_StopLoss.csv']
    labels = ['market_buy', 'market_sell', 'SL']
    stats = { label:get_stats_for_file(base_path + file_name) for label,file_name in zip(labels,file_names) }
    return stats'''

'''def get_slips_stats():
    buy = read_csv(ROOT_DIR+'/settings/slippages_market_buy.csv', header=0)
    sell = read_csv(ROOT_DIR+'/settings/slippages_market_sell.csv', header=0)
    SL = read_csv(ROOT_DIR+'/settings/slippages_StopLoss.csv', header=0)
    return {'market_buy':(float(np.mean(buy)), float(np.std(buy))),
            'market_sell':(float(np.mean(sell)), float(np.std(sell))),
            'SL':(float(np.mean(SL)), float(np.std(SL)))}'''


def get_slips_stats_advanced():
    buy = read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    sell = read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    SL = read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    return {
        'market_buy': {
            'mean': buy.mean(),
            'std': buy.std(),
            'skewness': buy.apply(skew),
            'kurtosis': buy.apply(kurtosis)
        },
        'market_sell': {
            'mean': sell.mean(),
            'std': sell.std(),
            'skewness': sell.apply(skew),
            'kurtosis': sell.apply(kurtosis)
        },
        'SL': {
            'mean': SL.mean(),
            'std': SL.std(),
            'skewness': SL.apply(skew),
            'kurtosis': SL.apply(kurtosis)
        }
    }


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
