import logging
from collections import deque
from csv import writer
from datetime import datetime as dt
from json import loads
from os import makedirs, path
from time import time

from binance.client import Client
from binance.enums import *
from binance.helpers import round_step_size
from binance.um_futures import UMFutures
from numpy import array, where, asarray
from websocket import WebSocketApp

from definitions import SLIPPAGE_DIR, LOG_DIR

ITV_AS_MS = {'1m': 60_000,
             '3m': 180_000,
             '5m': 300_000,
             '15m': 900_000,
             '30m': 1_800_000,
             '1h': 3_600_000,
             '2h': 7_200_000,
             '4h': 14_400_000,
             '6h': 21_600_000,
             '8h': 28_800_000,
             '12h': 43_200_000,
             '1d': 86_400_000,
             '3d': 259_200_000,
             '1w': 604_800_000,
             '1M': 2_419_200_000}

# Data from:
# https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
POSITION_TIER = {1: (125, .0040, 0), 2: (100, .005, 50),
                 3: (50, .01, 2_550), 4: (20, .025, 122_550),
                 5: (10, .05, 1_372_550), 6: (5, .10, 5_372_550),
                 7: (4, .125, 7_872_550), 8: (3, .15, 10_872_550),
                 9: (2, .25, 30_872_550), 10: (1, .50, 105_872_550)}


class SpotTaker:
    def __init__(self, base='BTC', quote='USDT', market='spot', itv='1m', settings=None, API_KEY='', SECRET_KEY='',
                 prev_size=100, multi=25):
        self.start_t = time()
        if settings is None:
            raise ValueError("Settings must be dict type.")
        self.base, self.quote = base, quote
        self.symbol = base + quote
        if ('enter_at' not in settings.keys()) or ('close_at' not in settings.keys()) or (
                'stop_loss' not in settings.keys()):
            raise AttributeError(
                "You should provide at least 'enter_at', 'close_at' and 'stop_loss' inside settings dict.")
        for key, value in settings.items():
            setattr(self, key, value)
        self.position_ratio /= 100 if self.position_ratio > 1 else self.position_ratio
        self.save_ratio /= 100 if self.save_ratio > 1 else self.save_ratio
        self.buy_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/market_buy.csv'
        self.sell_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/market_sell.csv'
        self.stoploss_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/limit_stop_loss.csv'
        makedirs(path.dirname(self.buy_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.sell_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.stoploss_slipp_file), exist_ok=True)

        url = f'wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp(url,
                               on_message=self.on_message,
                               on_error=self.on_error,
                               on_close=self.on_close,
                               on_open=self.on_open)
        self.client = Client(API_KEY, SECRET_KEY)
        prev_candles = self.client.get_historical_klines(self.symbol,
                                                         itv,
                                                         str(prev_size * multi) + " minutes ago UTC")
        prev_data = array([array(list(map(float, candle[1:6]))) for candle in prev_candles[:-1]])
        prev_data[where(prev_data[:, -2] == 0.0), -2] = 0.00000001
        self.OHLCV_data = deque(prev_data,
                                maxlen=len(prev_candles[:-1]))
        self.close = self.OHLCV_data[-1][3]
        self.init_balance = float(self.client.get_asset_balance(asset=self.quote)['free'])
        self.q = str(self.client.get_asset_balance(asset=self.base)['free'])[:7]
        self.balance = self.init_balance
        self.signal = 0.0
        self.cum_pnl = 0.0
        self.SL_placed = False
        self.stoploss_price = 0.0
        self.buy_order, self.sell_order, self.SL_order = None, None, None

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")
        return 0

    def on_open(self, ws):
        print("### WebSocket opened ###")

    def on_message(self, ws, message):
        # self.on_message_t = time()
        data_k = loads(message)['k']
        if data_k['x']:
            self.close = float(data_k['c'])
            # Used only with smaller timeframes like 1s
            fixed_volume = 0.00000001 if float(data_k['v']) <= 0.0 else float(data_k['v'])
            self.OHLCV_data.append(
                array(list(map(float, [data_k['o'], data_k['h'], data_k['l'], self.close, fixed_volume]))))
            self._analyze()
            if self.balance >= 5.01:
                self.cum_pnl = self.balance - self.init_balance
            print(f'INFO close:{self.close:.2f} bal:${self.balance:.2f} q:{self.q}', end=' ')
            print(f'cum_pnl:${self.cum_pnl:.2f}')
        if float(data_k['l']) <= self.stoploss_price:
            _order = self.client.get_order(symbol=self.symbol, orderId=self.SL_order['orderId'])
            if _order['status'] == 'FILLED':
                self.SL_placed = False
                self.stoploss_price = 0.0
                self.balance = float(self.client.get_asset_balance(asset=self.quote)['free'])
                self.q = '0.00000'
                self._report_slipp(_order, self.stoploss_price, 'stoploss')
            else:
                self._partially_filled_problem()
        if time() - self.start_t >= 86_340:
            self.ws.close()
            self.start_t = time()
            self.ws.run_forever()

    def _analyze(self):
        # print(f'(on_message to _analyze: {(time() - self.on_message_t) / 1_000}ms)')
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
        self.signal = 0.0
        print(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')

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
        print(f' {dt.today()} REPORTING {order_type} SLIP {_slipp}')

    def _get_filled_price(self, order):
        value, quantity = 0, 0
        for element in order['fills']:
            qty = element['qty']
            value += float(element['price']) * float(qty)
            quantity += float(qty)
        return value / quantity

    def _partially_filled_problem(self):
        self._cancel_all_orders()
        self.q = str(self.client.get_asset_balance(asset=self.base)['free'])[:7]
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
            order_t = time()
            self.buy_order = self.client.order_market_buy(symbol=self.symbol,
                                                          quantity=q)
            print(f'(sending buy order: {(time() - order_t) * 1_000}ms)')
            self.q = q
            self.balance = float(self.client.get_asset_balance(asset=self.quote)['free'])
            print(f' {dt.today()} BUY_MARKET q:{q} self.balance:{self.balance} self.q:{self.q}')
            return True
        except Exception as e:
            print(f'exception at _market_buy(): {e}')
            return False
        # print(self.buy_order)

    def _market_sell(self, q):
        try:
            order_t = time()
            self.sell_order = self.client.order_market_sell(symbol=self.symbol,
                                                            quantity=q)
            print(f'(sending sell order: {(time() - order_t) * 1_000}ms)')
            print(f'(_analyze to _market_sell: {(time() - self.analyze_t) * 1_000}ms)')
            self.balance = float(self.client.get_asset_balance(asset=self.quote)['free'])
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
class FuturesTaker:
    def __init__(self, base='BTC', quote='USDT', market='um', itv='1m', settings=None, API_KEY='', SECRET_KEY='',
                 prev_size=100, multi=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        file_handler = logging.FileHandler(
            f'{LOG_DIR}{dt.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.__class__.__name__}.log')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

        self.start_t = time()
        if settings is None:
            raise ValueError("Settings must be dict type.")
        else:
            logging.info(f'Settings provided: {settings}')
        self.base, self.quote = base, quote
        self.symbol = base + quote
        for key, value in settings.items():
            setattr(self, key, value)
        self.position_ratio /= 100 if self.position_ratio > 1 else self.position_ratio
        self.save_ratio /= 100 if self.save_ratio > 1 else self.save_ratio
        # Slippage reporting files
        self.buy_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/market_buy.csv'
        self.sell_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/market_sell.csv'
        self.stoploss_slipp_file = f'{SLIPPAGE_DIR}{market}/{self.symbol}{itv}/limit_stop_loss.csv'
        makedirs(path.dirname(self.buy_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.sell_slipp_file), exist_ok=True)
        makedirs(path.dirname(self.stoploss_slipp_file), exist_ok=True)

        url = f'wss://fstream.binance.com/ws/{self.symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp(url,
                               on_message=self.on_message,
                               on_error=self.on_error,
                               on_close=self.on_close,
                               on_open=self.on_open)
        self.client = UMFutures(API_KEY, SECRET_KEY)

        self.OHLCV_data = self._collect_previous_candles(itv, prev_size, multi)
        self.close = self.OHLCV_data[-1][3]

        # Balance
        self.available_balance = self._get_available_balance(self.quote)
        self.init_balance = self.available_balance
        self.position_balance = self.trade_balance * self.position_ratio
        if self.trade_balance > self.available_balance:
            raise RuntimeError(
                f'Account does not have required quote balance. Available: {self.available_balance}, required:{self.trade_balance}')
        self.save_balance = self.available_balance - self.trade_balance
        self.pre_trade_balance = self.trade_balance
        self.q = 0.0
        self._check_tier()

        self.orders = self.client.get_orders(symbol=self.symbol)
        # Leverage change
        try:
            self.client.change_leverage(symbol=self.symbol,
                                        leverage=str(self.leverage))
            self.logger.info(f'Leverage changed to: {self.leverage}x')
        except Exception as e:
            self.logger.error(f'Changing leverage error: {e}')
        # Margin type change
        try:
            m_type = self.client.change_margin_type(symbol=self.symbol,
                                                    marginType="ISOLATED")
        except Exception as e:
            if e.args[2] == 'No need to change margin type.':
                self.logger.info('Margin type is already set to ISOLATED')
            else:
                self.logger.info(f'Changing margin type error: {e}')

        for s in self.client.exchange_info()["symbols"]:
            if s['symbol'] == self.symbol:
                self.tick_size = float(s["filters"][0]['tickSize'])
                self.step_size = float(s["filters"][1]['stepSize'])
                self.min_qty = float(s["filters"][1]['minQty'])
                self.max_qty = float(s["filters"][1]['maxQty'])

        self.signal = 0.0
        self.cum_pnl = 0.0
        self.SL_placed = False
        self.in_long_position, self.in_short_position = False, False
        self.stoploss_price, self.takeprofit_price = 0.0, 0.0
        self.buy_order, self.sell_order, self.SL_order, self.TP_order = None, None, None, None
        self.logger.debug(f'prev_data[-5:]: {asarray(self.OHLCV_data)[-5:, :]}, len: {len(self.OHLCV_data)}')
        self.logger.info(f'Last close price: ${self.close}')
        self.logger.info(
            f'per_trade_bal:${self.position_balance:.2f} trade_available_bal:${self.trade_balance:.2f} save_bal:${self.save_balance:.2f} full_bal:${self.available_balance:.2f}')
        self.logger.info(
            f'step_size: {self.step_size} min_qty: {self.min_qty} max_qty: {self.max_qty} per_trade_qty: {(self.position_balance * self.leverage) / self.close}')
        self.logger.info(f'last orders {self.orders}')

    def on_error(self, ws, error):
        self.logger.error(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.debug("### WebSocket closed ###")

    def on_open(self, ws):
        self.logger.debug("### WebSocket opened ###")

    def on_message(self, ws, message):
        # self.on_message_t = time()
        data_k = loads(message)['k']
        if data_k['x']:
            self.close = float(data_k['c'])
            # Used only with smaller timeframes like 1s
            fixed_volume = 0.00000001 if float(data_k['v']) <= 0.0 else float(data_k['v'])
            self.OHLCV_data.append(
                array(list(map(float, [data_k['o'], data_k['h'], data_k['l'], self.close, fixed_volume]))))
            self._analyze()
            if (not self.in_long_position) and (not self.in_short_position):
                self.cum_pnl = self.available_balance - self.init_balance
            self.logger.info(
                f'close:{self.close:.2f} q:{self.q} per_trade_bal:${self.position_balance:.2f} trade_available_bal:${self.trade_balance:.2f} save_bal:${self.save_balance:.2f} full_bal:${self.available_balance:.2f} cum_pnl:${self.cum_pnl:.2f}')
        # Stop Loss filling handle
        if self.SL_order is not None:
            if ((float(data_k['l']) <= self.stoploss_price) and self.in_long_position) or (
                    (float(data_k['h']) >= self.stoploss_price) and self.in_short_position):
                order = self.client.query_order(symbol=self.symbol, orderId=self.SL_order['orderId'])
                self._cancel_all_orders()
                if order['status'] != 'FILLED':
                    self.logger.warning(f'STOP_LOSS was not filled. orderID:{self.SL_order["orderId"]}')
                    self._partially_filled_problem(self.stoploss_price)
                else:
                    self.logger.info(f'STOP_LOSS was filled.')
                self._update_balances()
                self.in_long_position, self.in_short_position = False, False
        if self.TP_order is not None:
            if ((float(data_k['h']) >= self.takeprofit_price) and self.in_long_position) or (
                    (float(data_k['l']) <= self.takeprofit_price) and self.in_short_position):
                order = self.client.query_order(symbol=self.symbol, orderId=self.TP_order['orderId'])
                self._cancel_all_orders()
                if order['status'] != 'FILLED':
                    self.logger.warning(f'TAKE_PROFIT was not filled. orderID:{self.TP_order["orderId"]}')
                    self._partially_filled_problem(self.takeprofit_price)
                else:
                    self.logger.info(f'TAKE_PROFIT was filled.')
                self._update_balances()
                self.in_long_position, self.in_short_position = False, False
        # Reopen websocket connection just to avoid timeout DC
        if time() - self.start_t >= 86_340:
            self.ws.close()
            self.start_t = time()
            self.ws.run_forever()

    def _analyze(self):
        # print(f'(on_message to _analyze: {(time() - self.on_message_t) / 1_000}ms)')
        self.analyze_t = time()
        self._check_signal()
        if self.in_long_position and (self.signal <= -self.long_close_at):
            if self._market_sell(self.q):
                self.in_long_position = False
                self._cancel_all_orders()
                self._report_slipp(self.sell_order, self.close, 'sell')
                self._update_balances()
        elif self.in_short_position and (self.signal >= self.short_close_at):
            if self._market_buy(self.q):
                self.in_short_position = False
                self._cancel_all_orders()
                self._report_slipp(self.buy_order, self.close, 'buy')
                self._update_balances()
        elif (not self.in_long_position) and (not self.in_short_position):
            if self.signal >= self.long_enter_at:
                trade_q = (self.position_balance * self.leverage) / self.close
                q = str(trade_q)[:len(str(self.step_size))]
                if self._market_buy(q):
                    self.stoploss_price = round_step_size(self.close * (1 - self.stop_loss), self.tick_size)
                    self._stop_loss(q, self.stoploss_price, 'SELL')
                    self.takeprofit_price = round_step_size(self.close * (1 + self.take_profit), self.tick_size)
                    self._take_profit(q, self.takeprofit_price, 'SELL')
                    self._report_slipp(self.buy_order, self.close, 'buy')
                    self.q = q
                    self.in_long_position = True
            elif self.signal <= -self.short_enter_at:
                trade_q = (self.position_balance * self.leverage) / self.close
                q = str(trade_q)[:len(str(self.step_size))]
                if self._market_sell(q):
                    self.stoploss_price = round_step_size(self.close * (1 + self.stop_loss), self.tick_size)
                    self._stop_loss(q, self.stoploss_price, 'BUY')
                    self.takeprofit_price = round_step_size(self.close * (1 - self.take_profit), self.tick_size)
                    self._take_profit(q, self.takeprofit_price, 'BUY')
                    self._report_slipp(self.sell_order, self.close, 'sell')
                    self.q = q
                    self.in_short_position = True

    def _check_signal(self):
        self.signal = 0.0
        self.logger.debug(f'(_analyze to _check_signal: {(time() - self.analyze_t) * 1_000}ms)')

    def _report_slipp(self, order, req_price, order_type):
        _order = self.client.query_order(symbol=self.symbol, orderId=order['orderId'])
        retry = 0
        while _order['status'] != 'FILLED':
            _order = self.client.query_order(symbol=self.symbol, orderId=order['orderId'])
            retry += 1
            if retry > 10:
                self.logger.debug(f'Order was not filled after {retry} tries at _report_slipp() {_order}')
                break
        if order_type == 'buy':
            file = self.buy_slipp_file
        elif order_type == 'sell':
            file = self.sell_slipp_file
        elif (order_type == 'stoploss') or (order_type == 'unfilled_stop'):
            file = self.stoploss_slipp_file
        else:
            self.logger.debug(f'order type= {order_type} not handled at _report_slipp')
            return
        _slipp = float(_order['avgPrice']) / req_price
        with open(file, 'a', newline='') as file:
            writer(file).writerow([_slipp])
        self.logger.info(f' {dt.today()} REPORTING {order_type} SLIP {_slipp}')

    def _get_filled_price(self, order):
        self.logger.debug(order)
        return 0
        # value, quantity = 0, 0
        # for element in order['fills']:
        #     qty = element['qty']
        #     value += float(element['price']) * float(qty)
        #     quantity += float(qty)
        # return value / quantity

    def _partially_filled_problem(self, req_price):
        if self.in_long_position:
            if self._market_sell(self.q):
                self._report_slipp(self.sell_order, req_price, 'unfilled_stop')
        elif self.in_short_position:
            if self._market_buy(self.q):
                self._report_slipp(self.buy_order, req_price, 'unfilled_stop')
        else:
            self.logger.error(f'FAILED AT _partially_filled_problem()')

    def _cancel_all_orders(self):
        try:
            self.client.cancel_open_orders(symbol=self.symbol)
            self.TP_order, self.SL_order = None, None
            self.takeprofit_price, self.stoploss_price = 0.0, 0.0
        except Exception as e:
            self.logger.error(f'exception(_cancel_all_orders): {e}')

    def _cancel_order_by_id(self, order_id):
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
        except Exception as e:
            self.logger.error(f'exception(_cancel_order): {e}')

    def _stop_loss(self, q, price, side):
        try:
            self.SL_order = self.client.new_order(symbol=self.symbol,
                                                  side=side,
                                                  type='STOP_MARKET',
                                                  # quantity=q,
                                                  stopPrice=price,
                                                  closePosition='true')
            self.logger.info(f'STOP_MARKET q:{q} stopPrice:{price} {dt.today()} ')
            return True
        except Exception as e:
            self.logger.error(f'exception at _stop_loss(): {e}')
            return False

    def _take_profit(self, q, price, side):
        try:
            self.TP_order = self.client.new_order(symbol=self.symbol,
                                                  side=side,
                                                  type='TAKE_PROFIT_MARKET',
                                                  # quantity=q,
                                                  stopPrice=price,
                                                  closePosition='true')
            self.logger.info(f'TAKE_PROFIT_MARKET q:{q} stopPrice:{price} {dt.today()} ')
            return True
        except Exception as e:
            self.logger.error(f'exception at _take_profit(): {e}')
            return False

    def _market_buy(self, q):
        try:
            order_t = time()
            self.buy_order = self.client.new_order(symbol=self.symbol,
                                                   side='BUY',
                                                   type='MARKET',
                                                   quantity=q)
            self.logger.debug(f'(sending buy order: {(time() - order_t) * 1_000}ms)')
            self.logger.debug(f'(_analyze to _market_buy: {(time() - self.analyze_t) * 1_000}ms)')
            self.logger.info(f'BUY_MARKET q:{q} position_balance:{self.position_balance} {dt.today()}')
            return True
        except Exception as e:
            self.logger.error(f'exception at _market_buy(): {e}')
            return False

    def _market_sell(self, q):
        try:
            order_t = time()
            self.sell_order = self.client.new_order(symbol=self.symbol,
                                                    side='SELL',
                                                    type='MARKET',
                                                    quantity=q)
            self.logger.debug(f'(sending sell order: {(time() - order_t) * 1_000}ms)')
            self.logger.debug(f'(_analyze to _market_sell: {(time() - self.analyze_t) * 1_000}ms)')
            self.logger.info(f'SELL_MARKET q:{q} position_balance:{self.position_balance} {dt.today()}{dt.today()}')
            return True
        except Exception as e:
            self.logger.error(f'exception at _market_sell(): {e}')
            return False

    def _collect_previous_candles(self, itv, prev_size, multi):
        current_server_time = self.client.time()['serverTime']
        _start_time = int(current_server_time - (prev_size * multi) * ITV_AS_MS[itv])
        self.logger.debug(
            f'Collecting {prev_size * multi} previous candles from {_start_time} to {current_server_time}')
        limit = 1_000

        # Fetch candles in segments of limit*ITV_AS_MS[itv]
        prev_candles = []
        while len(prev_candles) < prev_size * multi:
            fetched_candles = self.client.klines(symbol=self.symbol,
                                                 interval=itv,
                                                 limit=limit,
                                                 startTime=_start_time)
            if not fetched_candles:
                break
            prev_candles.extend(fetched_candles)
            # Moving forward in time
            _start_time += limit * ITV_AS_MS[itv]

        # [:-1] removing last value as the candle has not closed yet.
        prev_data = array([array(list(map(float, candle[1:6]))) for candle in prev_candles[:-1]])
        # Fixing volume if zero.
        prev_data[where(prev_data[:, -2] == 0.0), -2] = 0.00000001
        return deque(prev_data,
                     maxlen=len(prev_candles[:-1]))

    def _get_available_balance(self, asset):
        for bal in self.client.balance():
            if bal['asset'] == asset:
                return float(bal['availableBalance'])

    def _update_balances(self):
        self.available_balance = self._get_available_balance(self.quote)
        gain = self.available_balance - self.save_balance - self.trade_balance
        if gain > 0:
            self.save_balance += gain * self.save_ratio
            self.trade_balance += gain * (1 - self.save_ratio)
            self.position_balance = self.trade_balance * self.position_ratio
        else:
            self.trade_balance += gain
            self.position_balance = self.trade_balance * self.position_ratio
        self.q = 0.0

    def _check_tier(self):
        # print('_check_tier')
        if self.trade_balance < 50_000:
            self.tier = 1
        elif 50_000 < self.trade_balance < 500_000:
            self.tier = 2
        elif 500_000 < self.trade_balance < 8_000_000:
            self.tier = 3
        elif 8_000_000 < self.trade_balance < 50_000_000:
            self.tier = 4
        elif 50_000_000 < self.trade_balance < 80_000_000:
            self.tier = 5
        elif 80_000_000 < self.trade_balance < 100_000_000:
            self.tier = 6
        elif 100_000_000 < self.trade_balance < 120_000_000:
            self.tier = 7
        elif 120_000_000 < self.trade_balance < 200_000_000:
            self.tier = 8
        elif 200_000_000 < self.trade_balance < 300_000_000:
            self.tier = 9
        elif 300_000_000 < self.trade_balance < 500_000_000:
            self.tier = 10
        if self.leverage > POSITION_TIER[self.tier][0]:
            self.logger.warning(f' Leverage exceeds tier {self.tier} max', end=' ')
            # print(f'changing from {self.leverage} to {self.POSITION_TIER[self.tier][0]} (Balance:${self.balance}:.2f)')
            self.leverage = POSITION_TIER[self.tier][0]

    def run(self):
        self.ws.run_forever()
