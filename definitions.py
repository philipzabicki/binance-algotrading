from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = ROOT_DIR + '/data/'
REPORT_DIR = ROOT_DIR + '/reports/'
SLIPPAGE_DIR = DATA_DIR + '/slippage/'
TENSORBOARD_DIR = ROOT_DIR + '/tensorboard/'
MODELS_DIR = ROOT_DIR + '/model/'

ADDITIONAL_DATA_BY_OHLCV_MA = {0: 1, 1: 7, 2: 1, 3: 6, 4: 1, 5: 8, 6: 9, 7: 11, 8: 2, 9: 1, 10: 1, 11: 19, 12: 1, 13: 9,
                               14: 21, 15: 19, 16: 1, 17: 1, 18: 1, 19: 6, 20: 1, 21: 1, 22: 15, 23: 63, 24: 1, 25: 15,
                               26: 1, 27: 1, 28: 1, 29: 15, 30: 1, 31: 4, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1}

ADDITIONAL_DATA_BY_MA = {0: 1, 1: 6, 2: 1, 3: 5, 4: 1, 5: 8, 6: 9, 7: 10, 8: 2, 9: 1, 10: 6, 11: 1, 12: 1, 13: 14,
                         14: 14, 15: 62, 16: 1, 17: 1, 18: 1, 19: 1, 20: 3, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1}
