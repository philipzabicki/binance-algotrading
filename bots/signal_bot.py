from collections import deque
from time import time

from numpy import array, where, asarray

from bots import SpotTakerBot
from utils.ta_tools import custom_MACD, MACD_cross_signal, get_MA_band_signal


class MACDSpotTakerBot(SpotTakerBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs['settings'].items():
            setattr(self, key, value)
        previous_size = max(self.slow_period, self.fast_period, self.signal_period) * kwargs['multi']
        prev_candles = self.client.get_historical_klines(self.symbol,
                                                         kwargs['itv'],
                                                         str(previous_size) + " minutes ago UTC")
        prev_data = array([array(list(map(float, candle[1:6]))) for candle in prev_candles[:-1]])
        prev_data[where(prev_data[:, -1] == 0.0), -1] = 0.00000001
        self.OHLCV_data = deque(prev_data,
                                maxlen=len(prev_candles[:-1]))
        self.close = self.OHLCV_data[-1][3]
        print(f'SETTINGS: {kwargs["settings"]}, prev_data_size:{len(self.OHLCV_data)}')
        print(
            f'Initial q:{self.q}, balance:{self.balance} last_{kwargs["itv"]}_close:{self.close} prev_data_size:{len(self.OHLCV_data)}')

    def _check_signal(self):
        self.macd, self.signal_line = custom_MACD(asarray(self.OHLCV_data),
                                                  fast_ma_type=self.fast_ma_type, fast_period=self.fast_period,
                                                  slow_ma_type=self.slow_ma_type, slow_period=self.slow_period,
                                                  signal_ma_type=self.signal_ma_type, signal_period=self.signal_period)
        # We only need last 2 values of macd and signal line to get trade signals
        self.signals = MACD_cross_signal(self.macd[-4:], self.signal_line[-4:])
        self.signal = self.signals[-1]
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    MACD:{self.macd[-3:]} SIGNAL_LINE:{self.signal_line[-3:]} trade_signals:{self.signals}')


class BandsSpotTakerBot(SpotTakerBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs['settings'].items():
            setattr(self, key, value)
        previous_size = max(self.ma_period, self.atr_period) * kwargs['multi']
        prev_candles = self.client.get_historical_klines(self.symbol,
                                                         kwargs['itv'],
                                                         str(previous_size) + " minutes ago UTC")
        prev_data = array([array(list(map(float, candle[1:6]))) for candle in prev_candles[:-1]])
        prev_data[where(prev_data[:, -1] == 0.0), -1] = 0.00000001
        self.OHLCV_data = deque(prev_data,
                                maxlen=len(prev_candles[:-1]))
        self.close = self.OHLCV_data[-1][3]
        print(f'SETTINGS: {kwargs["settings"]}, prev_data_size:{len(self.OHLCV_data)}')
        print(
            f'Initial q:{self.q}, balance:{self.balance} last_{kwargs["itv"]}_close:{self.close} prev_data_size:{len(self.OHLCV_data)}')

    def _check_signal(self):
        self.OHLCV0_np = array(self.OHLCV_data)
        self.signals = get_MA_band_signal(asarray(self.OHLCV_data),
                                          self.ma_type, self.ma_period,
                                          self.atr_period, self.atr_multi)
        self.signal = self.signals[-1]
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')
        print(f'    signals:{self.signals[-3:]}')
