from time import time

from numpy import asarray
from talib import RSI, ATR, AD

from bots import SpotTaker
from utils.ta_tools import custom_MACD, MACD_cross_signal, anyMA_sig, get_MA, RSI_like_signal, custom_ChaikinOscillator, \
    ChaikinOscillator_signal


class MACDSignalsBot(object):
    def __init__(self, *args, bot_type=SpotTaker, **kwargs):
        self.__class__ = type(self.__class__.__name__,
                              (bot_type, object),
                              dict(self.__class__.__dict__))
        previous_size = max(kwargs['settings']['slow_period'], kwargs['settings']['fast_period']) + kwargs['settings']['signal_period']
        super(self.__class__, self).__init__(*args, prev_size=previous_size, **kwargs)

    def _check_signal(self):
        self.macd, self.signal_line = custom_MACD(asarray(self.OHLCV_data),
                                                  fast_ma_type=self.fast_ma_type, fast_period=self.fast_period,
                                                  slow_ma_type=self.slow_ma_type, slow_period=self.slow_period,
                                                  signal_ma_type=self.signal_ma_type, signal_period=self.signal_period)
        # We only need last 2 values of macd and signal line to get trade signals
        self.signals = MACD_cross_signal(self.macd[-3:], self.signal_line[-3:])
        self.signal = self.signals[-1]
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    MACD:{self.macd[-3:]} SIGNAL_LINE:{self.signal_line[-3:]} trade_signals:{self.signals}')


class BandsSignalsBot(object):
    def __init__(self, *args, bot_type=SpotTaker, **kwargs):
        self.__class__ = type(self.__class__.__name__,
                              (bot_type, object),
                              dict(self.__class__.__dict__))
        previous_size = max(kwargs['settings']['ma_period'], kwargs['settings']['atr_period'])
        super(self.__class__, self).__init__(*args, prev_size=previous_size, **kwargs)

    def _check_signal(self):
        self.ma = get_MA(asarray(self.OHLCV_data),
                         self.ma_type, self.ma_period)
        # *asarray(self.OHLCV_data)[:, 1:4].T translates as:
        #  self.OHLCV_data)[:, 1], self.OHLCV_data)[:, 2], self.OHLCV_data)[:, 3]
        self.atr = ATR(*asarray(self.OHLCV_data)[:, 1:4].T, self.atr_period)
        self.signals = anyMA_sig(asarray(self.OHLCV_data)[-3:, 3], self.ma[-3:], self.atr[-3:], self.atr_multi)
        self.signal = self.signals[-1]
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    MA:{self.ma[-3:]} ATR:{self.atr[-3:]} trade_signals:{self.signals}')


class ChaikinOscillatorSignalsBot(object):
    def __init__(self, *args, bot_type=SpotTaker, **kwargs):
        self.__class__ = type(self.__class__.__name__,
                              (bot_type, object),
                              dict(self.__class__.__dict__))
        previous_size = max(kwargs['settings']['slow_period'], kwargs['settings']['fast_period'])
        super(self.__class__, self).__init__(*args, prev_size=previous_size, **kwargs)

    def _check_signal(self):
        # Can be done faster as adl does not require as much lookback data
        self.adl = AD(*asarray(self.OHLCV_data)[:, 1:5].T)
        self.chaikin_oscillator = custom_ChaikinOscillator(self.adl, fast_ma_type=self.fast_ma_type, fast_period=self.fast_period,
                                                           slow_ma_type=self.slow_ma_type, slow_period=self.slow_period)
        self.signals = ChaikinOscillator_signal(self.chaikin_oscillator[-3:])
        self.signal = self.signals[-1]
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    ChaikinOscillator:{self.chaikin_oscillator[-3:]} trade_signals:{self.signals}')


class MACDRSISignalsBot(object):
    def __init__(self, *args, bot_type=SpotTaker, **kwargs):
        self.__class__ = type(self.__class__.__name__,
                              (bot_type, object),
                              dict(self.__class__.__dict__))
        previous_size = max((max(kwargs['settings']['slow_period'], kwargs['settings']['fast_period']) + kwargs['settings']['signal_period']),
                            kwargs['settings']['rsi_period'])
        super(self.__class__, self).__init__(*args, prev_size=previous_size, **kwargs)

    def _check_signal(self):
        self.macd, self.signal_line = custom_MACD(asarray(self.OHLCV_data),
                                                  fast_ma_type=self.fast_ma_type, fast_period=self.fast_period,
                                                  slow_ma_type=self.slow_ma_type, slow_period=self.slow_period,
                                                  signal_ma_type=self.signal_ma_type, signal_period=self.signal_period)
        self.rsi = RSI(asarray(self.OHLCV_data)[:, 3], self.rsi_period)
        # We only need last 2 values of macd and signal line to get trade signals
        self.signals1 = MACD_cross_signal(self.macd[-3:], self.signal_line[-3:])
        self.signals2 = RSI_like_signal(self.rsi[-3:], self.rsi_period)
        self.signal = (self.signals1[-1], self.signals2[-1])
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    MACD:{self.macd[-3:]} SIGNAL_LINE:{self.signal_line[-3:]}')
        print(f'    signals1:{self.signals1} signals1:{self.signals1} signal:{self.signal}')
        raise NotImplementedError("More than single signal trading not implemented yet.")