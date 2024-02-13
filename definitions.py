from os import path

ROOT_DIR = path.dirname(path.abspath(__file__))
DATA_DIR = ROOT_DIR + '/data/'
REPORT_DIR = ROOT_DIR + '/reports/'
SLIPPAGE_DIR = DATA_DIR + '/slippage/'
TENSORBOARD_DIR = ROOT_DIR + '/tensorboard/'
MODELS_DIR = ROOT_DIR + '/model/'

ADDITIONAL_DATA_BY_OHLCV_MA = {0: 1, 1: 4, 2: 1, 3: 3, 4: 1, 5: 5, 6: 6, 7: 7, 8: 1, 9: 1, 10: 1, 11: 14, 12: 1, 13: 7, 14: 16, 15: 14, 16: 1, 17: 1, 18: 1, 19: 3, 20: 1, 21: 1, 22: 9, 23: 1, 24: 1, 25: 10, 26: 1, 27: 1, 28: 1, 29: 9, 30: 1, 31: 2, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1}

ADDITIONAL_DATA_BY_MA = {0: 1, 1: 4, 2: 1, 3: 3, 4: 1, 5: 5, 6: 6, 7: 7, 8: 1, 9: 1, 10: 3, 11: 1, 12: 1, 13: 9, 14: 10, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 2, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1}
