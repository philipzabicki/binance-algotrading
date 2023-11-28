from websocket import WebSocketApp
from binance.client import Client
from json import loads
from numpy import array, where
from os import getcwd
from csv import writer
from collections import deque
from time import time
from datetime import datetime as dt
from talib import RSI
from TA_tools import RSI_like_signal
from definitions import SETTINGS_DIR
from binance.enums import *


class TakerBot:
    def __init__(self, symbol, itv, settings, API_KEY, SECRET_KEY, multi=25):
        self.buy_slipp_file = SETTINGS_DIR + 'slippages_market_buy.csv'
        self.symbol = symbol
        self.settings = settings
        url = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp(url,
                               on_message=self.on_message,
                               on_error=self.on_error,
                               on_close=self.on_close,
                               on_open=self.on_open)
        self.client = Client(API_KEY, SECRET_KEY)
        print(f'SETTINGS: {self.settings}')
        prev_candles = self.client.get_historical_klines(symbol,
                                                         itv,
                                                         str(self.settings['period'] * multi) + " minutes ago UTC")
        prev_data = array([list(map(float, candle[1:6])) for candle in prev_candles[:-1]])
        self.OHLCVX_data = deque(prev_data,
                                 maxlen=len(prev_candles[:-1]))
        # print(self.OHLCVX_data)
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        self.buy_amount = self.settings['buy_amount']
        self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
        print(f'QUANTITY: {self.q} BALANCE:{self.balance}')
        self.rsi, self.signal = 0.0, 0.0
        self.buy_order = None

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")

    def on_open(self, ws):
        print("### WebSocket opened ###")

    def on_message(self, ws, message):
        # self.start_t1 = time()
        data = loads(message)
        #print(f"Received: {data}")
        data_k = data['k']
        if data_k['x']:
            self.OHLCVX_data.append(
                list(map(float, [data_k['o'], data_k['h'], data_k['l'], data_k['c'], data_k['v']])))
            self._analyze(float(data_k['c']))
            print(f' INFO close:{data_k["c"]} rsi:{self.rsi[-1]} signal:{self.signal[-1]} qty:{self.q} balance:{self.balance}', end=' ')
            # print(f'init_balance:{self.init_balance:.2f}')

    def _analyze(self, close):
        # print(f'(on_message to _analyze: {time()-self.start_t1}s)')
        # self.start_t2 = time()
        self._check_signal(close)
        if (self.signal == self.settings['enter_at']) and (self.balance >= self.buy_amount):
            q = str((self.buy_amount / close) + .00001)[:7]
            self._market_buy(q)
            self._report_slipp(self.buy_order, close, 'buy')

    def _check_signal(self, close):
        self.OHLCV0_np = array(self.OHLCVX_data)
        self.rsi = RSI(self.OHLCV0_np[:, 3], timeperiod=self.settings['period'])
        self.signal = RSI_like_signal(self.rsi, self.settings['period'])
        # print(f'(_analyze to _check_signal: {time()-self.start_t2}s)')

    def _report_slipp(self, order, req_price, order_type):
        file = self.buy_slipp_file
        if order_type == 'buy':
            file = self.buy_slipp_file
        elif order_type == 'sell':
            file = self.sell_slipp_file
        elif order_type == 'stoploss':
            file = self.stoploss_slipp_file
            with open(file, 'a', newline='') as file:
                writer(file).writerow([float(order['price']) / req_price])
            return
        # print(f'REPORTING {order_type} SLIPP FROM: {order}')
        print(f'{dt.today()} REPORTING {order_type} SLIPP')
        filled_price = self._get_filled_price(order)
        with open(file, 'a', newline='') as file:
            writer(file).writerow([filled_price / req_price])

    def _get_filled_price(self, order):
        value, quantity = 0, 0
        for element in order['fills']:
            qty = element['qty']
            value += float(element['price']) * float(qty)
            quantity += float(qty)
        return value / quantity

    def _market_buy(self, q):
        try:
            self.buy_order = self.client.order_market_buy(symbol=self.symbol,
                                                          quantity=q)
            # print(f'(_analyze to _market_buy: {time() - self.start_t2}s)')
            self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            print(f'{dt.today()} BUY_MARKET q:{q}')
            return self.buy_order
        except Exception as e:
            print(f'exception(_market_buy): {e}')
        # print(self.buy_order)

    def run(self):
        self.ws.run_forever()
