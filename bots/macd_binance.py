from websocket import WebSocketApp
from binance.client import Client
from json import loads
from numpy import array, where, asarray
from os import makedirs, path
from csv import writer
from collections import deque
from time import time
from datetime import datetime as dt
from binance.enums import *
from TA_tools import custom_MACD, MACD_cross_signal
from definitions import SLIPPAGE_DIR


class TakerBot:
    def __init__(self, symbol, market, itv, settings, API_KEY, SECRET_KEY, multi=25):
        for key, value in settings.items():
            setattr(self, key, value)
        self.buy_slipp_file = f'{SLIPPAGE_DIR}{market}/{symbol}{itv}/market_buy.csv'
        self.sell_slipp_file = f'{SLIPPAGE_DIR}{market}/{symbol}{itv}/market_sell.csv'
        self.stoploss_slipp_file = f'{SLIPPAGE_DIR}{market}/{symbol}{itv}/limit_stop_loss.csv'
        makedirs(path.dirname(self.buy_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.sell_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.stoploss_slipp_file), exist_ok=True)

        self.symbol = symbol
        url = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp(url,
                               on_message=self.on_message,
                               on_error=self.on_error,
                               on_close=self.on_close,
                               on_open=self.on_open)
        self.client = Client(API_KEY, SECRET_KEY)
        prev_candles = self.client.get_historical_klines(symbol,
                                                         itv,
                                                         str(max(self.slow_period,
                                                                 self.fast_period,
                                                                 self.signal_period) * multi) + " minutes ago UTC")
        prev_data = array([array(list(map(float, candle[1:6]))) for candle in prev_candles[:-1]])
        prev_data[where(prev_data[:, -2] == 0.0), -2] = 0.00000001
        self.OHLCV_data = deque(prev_data,
                                 maxlen=len(prev_candles[:-1]))
        self.close = self.OHLCV_data[-1][3]
        self.init_balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        self.balance = self.init_balance
        self.signal = 0.0
        self.cum_pnl = 0.0
        self.SL_placed = False
        self.stoploss_price = 0.0
        self.buy_order, self.sell_order, self.SL_order = None, None, None
        print(f'SETTINGS: {settings}, prev_data_size:{len(self.OHLCV_data)}')
        print(f'Initial q:{self.q}, balance:{self.balance} last_{itv}_close:{self.close} prev_data_size:{len(self.OHLCV_data)}')

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")

    def on_open(self, ws):
        print("### WebSocket opened ###")

    def on_message(self, ws, message):
        self.on_message_t = time()
        data_k = loads(message)['k']
        if data_k['x']:
            self.close = float(data_k['c'])
            # Used only with smaller timeframes like 1s
            # fixed_volume = 0.00000001 if float(data_k['v']) <= 0.0 else float(data_k['v'])
            self.OHLCV_data.append(array(list(map(float, [data_k['o'], data_k['h'], data_k['l'], self.close, data_k['v']]))))
            self._analyze()
            if self.balance >= 5.01:
                self.cum_pnl = self.balance - self.init_balance
            print(f'INFO close:{self.close:.2f} bal:${self.balance:.2f} q:{self.q}', end=' ')
            print(f'macd:{self.macd[-3:]} sig_line:{self.signal_line[-3:]} sig:{self.signal}', end=' ')
            print(f'cum_pnl:${self.cum_pnl:.2f}')
        if float(data_k['l']) <= self.stoploss_price:
            _order = self.client.get_order(symbol=self.symbol, orderId=self.SL_order['orderId'])
            if _order['status'] == 'FILLED':
                self.SL_placed = False
                self.stoploss_price = 0.0
                self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
                self.q = '0.00000'
                self._report_slipp(_order, self.stoploss_price, 'stoploss')
            else:
                self._partially_filled_problem()

    def _analyze(self):
        print(f'(on_message to _analyze: {(time()-self.on_message_t/1_000)}ms)')
        self.analyze_t = time()
        self._check_signal()
        if (self.signal >= self.enter_at) and (self.balance > 5.01):
            q = str((self.balance / self.close) - .00001)[:7]
            if self._market_buy(q):
                self.stoploss_price = round(self.close * (1 - self.stop_loss), 2)
                self._stop_loss(q, self.stoploss_price)
                self._report_slipp(self.buy_order, self.close, 'buy')
        elif (self.signal <= -self.close_at) and (self.balance < 5.01):
            try:
                self._cancel_order(self.SL_order['orderId'])
            except Exception as e:
                self._cancel_all_orders()
                print(f'exception at _cancel_order(): {e}')
            if self._market_sell(self.q):
                self._report_slipp(self.sell_order, self.close, 'sell')
        # else:
        #     self.init_balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
        #     self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]

    def _check_signal(self):
        self.macd, self.signal_line = custom_MACD( asarray(self.OHLCV_data),
                                                   fast_ma_type=self.fast_ma_type, fast_period=self.fast_period,
                                                   slow_ma_type=self.slow_ma_type, slow_period=self.slow_period,
                                                   signal_ma_type=self.signal_ma_type, signal_period=self.signal_period)
        self.signal = MACD_cross_signal(self.macd, self.signal_line)[-1]
        print(f'(_analyze to _check_signal: {(time()-self.analyze_t)*1_000}ms)')

    def _report_slipp(self, order, req_price, order_type):
        file = self.buy_slipp_file
        if order_type == 'buy':
            file = self.buy_slipp_file
        elif order_type == 'sell':
            file = self.sell_slipp_file
        elif (order_type == 'stoploss') or (order_type == 'unfilled_stoploss'):
            file = self.stoploss_slipp_file
        # print(f'REPORTING {order_type} SLIPP FROM: {order}')
        filled_price = self._get_filled_price(order)
        _slipp = filled_price / req_price
        with open(file, 'a', newline='') as file:
            writer(file).writerow([_slipp])
        print(f'{dt.today()} REPORTING {order_type} SLIP {_slipp}')

    def _get_filled_price(self, order):
        value, quantity = 0, 0
        for element in order['fills']:
            qty = element['qty']
            value += float(element['price']) * float(qty)
            quantity += float(qty)
        return value / quantity

    def _partially_filled_problem(self):
        self._cancel_all_orders()
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        if self._market_sell(self.q):
            self._report_slipp(self.sell_order, self.stoploss_price, 'unfilled_stoploss')
            self.SL_placed = False
            self.stoploss_price = 0.0
        else:
            print(f'FAILED AT _partially_filled_problem()')

    def _cancel_all_orders(self):
        try:
            self.client._delete('openOrders', True, data={'symbol': self.symbol})
        except Exception as e:
            print(f'exception(_cancel_all_orders): {e}')

    def _cancel_order(self, order_id):
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
        except Exception as e:
            print(f'exception(_cancel_order): {e}')

    def _stop_loss(self, q, price):
        try:
            self.SL_order = self.client.create_order(symbol=self.symbol,
                                                     side=SIDE_SELL,
                                                     type=ORDER_TYPE_STOP_LOSS_LIMIT,
                                                     timeInForce=TIME_IN_FORCE_GTC,
                                                     quantity=q,
                                                     stopPrice=price,
                                                     price=price)
            self.SL_placed = True
            print(f' {dt.today()} STOPLOSS_LIMIT q:{q} price:{price}')
            return True
        except Exception as e:
            print(f'exception at _stop_loss(): {e}')
            return False

    def _market_buy(self, q):
        try:
            self.buy_order = self.client.order_market_buy(symbol=self.symbol,
                                                          quantity=q)
            print(f'(_analyze to _market_buy: {(time() - self.analyze_t)*1_000}ms)')
            self.q = q
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            print(f' {dt.today()} BUY_MARKET q:{q} self.balance:{self.balance} self.q:{self.q}')
            return True
        except Exception as e:
            print(f'exception at _market_buy(): {e}')
            return False
        # print(self.buy_order)

    def _market_sell(self, q):
        try:
            self.sell_order = self.client.order_market_sell(symbol=self.symbol,
                                                            quantity=q)
            # print(self.sell_order)
            print(f'(_analyze to _market_sell: {(time() - self.analyze_t)*1_000}ms)')
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            self.q = '0.00000'
            print(f' {dt.today()} SELL_MARKET q:{q} self.balance:{self.balance} self.q:{self.q}')
            return True
        except Exception as e:
            print(f'exception at _market_sell(): {e}')
            return False

    def run(self):
        self.ws.run_forever()


######################################################################################################################
######################################################################################################################
######################################################################################################################

class MakerBot:
    pass
