from websocket import WebSocketApp
from binance.client import Client
from json import loads
from numpy import array, where
from os import getcwd
from csv import writer
from collections import deque
from time import time
from datetime import datetime as dt
from talib import ATR
from binance.enums import *
from TA_tools import get_MA
from definitions import SETTINGS_DIR

class TakerBot:
    def __init__(self, symbol, itv, settings, API_KEY, SECRET_KEY, multi=25):
        self.buy_slipp_file = SETTINGS_DIR+'slippages_market_buy.old2.csv'
        self.sell_slipp_file = SETTINGS_DIR+'slippages_market_sell.old2.csv'
        self.stoploss_slipp_file = SETTINGS_DIR+'slippages_StopLoss.csv'
        #sl_slipp_file = '/settings/slippages_StopLoss.csv'
        #with open(self.cwd+self.buy_slipp_file, 'a', newline='') as file: self.buy_slipp_wr = writer(file)
        #with open(self.cwd+self.sell_slipp_file, 'a', newline='') as file: self.sell_slipp_wr = writer(file)
        #with open(cwd+sl_slipp_file, 'a', newline='') as file: self.sl_slipp_wr = writer(file)

        self.symbol = symbol
        url = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp(url,
                               on_message=self.on_message,
                               on_error=self.on_error,
                               on_close=self.on_close,
                               on_open=self.on_open)
        self.client = Client(API_KEY, SECRET_KEY)

        self.settings = settings
        print(f'SETTINGS: {self.settings}')
        prev_candles = self.client.get_historical_klines(symbol,
                                                         itv,
                                                         str(max(self.settings['MA_period'],
                                                                 self.settings['ATR_period']) * multi)+" seconds ago UTC")
        prev_data = array([list(map(float, candle[1:6]+[0])) for candle in prev_candles[:-1]])
        prev_data[where(prev_data[:, -2] == 0.0), -2] = 0.00000001
        self.OHLCVX_data = deque(prev_data,
                                 maxlen=len(prev_candles[:-1]))
        self.init_balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        self.balance = self.init_balance
        print(f'q={self.q} balance={self.balance}')
        self.signal = 0.0
        self.SL_placed = False
        self.stoploss_price = 0.0
        self.buy_order, self.sell_order, self.SL_order = None, None, None

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")
    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")
    def on_open(self, ws):
        print("### WebSocket opened ###")
    def on_message(self, ws, message):
        self.start_t1 = time()
        data = loads(message)
        #print(f"Received: {data}")
        data_k = data['k']
        if data_k['x']:
            _volume = '0.00000001' if float(data_k['v'])<=0.0 else data_k['v']
            self.OHLCVX_data.append(list( map(float, [data_k['o'],data_k['h'],data_k['l'],data_k['c'],_volume,0]) ))
            self._analyze(float(data_k['c']))
            #print(f' INFO close:{data_k["c"]} macd.py:{self.macd.py:.3f} balance:{self.balance:.2f} self.q:{self.q}', end=' ')
            #print(f'init_balance:{self.init_balance:.2f}')
        if float(data_k['l'])<=self.stoploss_price:
            _order = self.client.get_order(symbol=self.symbol, orderId=self.SL_order['orderId'])
            if _order['status']=='FILLED':
                self.SL_placed = False
                self.stoploss_price = 0.0
                self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
                self.q = '0.00000'
                self._report_slipp(_order, self.stoploss_price, 'stoploss')
            else:
                self._partialy_filled_problem()

    def _analyze(self, close):
        #print(f'(on_message to _analyze: {time()-self.start_t1}s)')
        self.start_t2 = time()
        self._check_signal(close)
        if (self.signal >= self.settings['enter_at']) and (self.balance>1.0):
            q = str((self.balance/close)-.00001)[:7]
            self._market_buy(q)
            self._report_slipp(self.buy_order, close, 'buy')
            self.stoploss_price = round(close*(1-self.settings['SL']),2)
            self._stop_loss(q, self.stoploss_price)
        elif (self.signal<=-self.settings['close_at']) and (self.balance<1.0):
            self._cancel_all_orders()
            self._market_sell(self.q)
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            self._report_slipp(self.sell_order, close, 'sell')

    def _check_signal(self, close):
        self.OHLCV0_np = array(self.OHLCVX_data)
        ma = get_MA(self.OHLCV0_np, self.settings['typeMA'], self.settings['MA_period'])
        atr = ATR(self.OHLCV0_np[:,1], self.OHLCV0_np[:,2], self.OHLCV0_np[:,3], timeperiod=self.settings['ATR_period'])
        self.signal = (ma[-1]-close)/(atr[-1]*self.settings['ATR_multi'])
        #print(f'(_analyze to _check_signal: {time()-self.start_t2}s)')
    
    def _report_slipp(self, order, req_price, order_type):
        file = self.buy_slipp_file
        if order_type=='buy':
            file = self.buy_slipp_file
        elif order_type=='sell':
            file = self.sell_slipp_file
        elif order_type=='stoploss':
            file = self.stoploss_slipp_file
            with open(file, 'a', newline='') as file:
                writer(file).writerow([float(order['price'])/req_price])
            return
        #print(f'REPORTING {order_type} SLIPP FROM: {order}')
        print(f'{dt.today()} REPORTING {order_type} SLIPP')
        filled_price = self._get_filled_price(order)
        with open(file, 'a', newline='') as file:
            writer(file).writerow([filled_price/req_price])

    def _get_filled_price(self, order):
        value, quantity = 0, 0
        for element in order['fills']:
            qty = element['qty']
            value += float(element['price'])*float(qty)
            quantity += float(qty)
        return value/quantity

    def _partialy_filled_problem(self):
        self.q = str(self.client.get_asset_balance(asset='BTC')['locked'])[:7]
        self._cancel_all_orders()
        _order = self._market_sell(self.q)
        self.SL_placed = False
        self.stoploss_price = 0.0
        print(f'{dt.today()} REPORTING stoploss(missed) SLIPP FROM: {_order}')
        file = self.stoploss_slipp_file
        with open(file, 'a', newline='') as file:
            writer(file).writerow([self._get_filled_price(_order)/self.stoploss_price])

    def _cancel_all_orders(self):
        try:
            self.client._delete('openOrders', True, data={'symbol': self.symbol})
            print(f'{dt.today()} DELETED ALL ORDERS')
        except Exception as e:  print(f'exception(_cancel_all_orders): {e}')
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
            print(f'{dt.today()} STOPLOSS_LIMIT q:{q} price:{price}')
            #print(self.SL_order)
        except Exception as e:
            print(f'exception at _stop_loss(): {e}')
    def _market_buy(self, q):
        try:
            self.buy_order = self.client.order_market_buy(  symbol=self.symbol,
                                                            quantity=q  )
            print(f'(_analyze to _market_buy: {time()-self.start_t2}s)')
            self.q = q
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            print(f'{dt.today()} BUY_MARKET q:{q}')
            return self.buy_order
        except Exception as e:
            print(f'exception at _market_buy(): {e}')
        #print(self.buy_order)
    def _market_sell(self, q):
        try:
            self.sell_order = self.client.order_market_sell(symbol=self.symbol,
                                                            quantity=q)
            #print(self.sell_order)
            print(f'(_analyze to _market_sell: {time()-self.start_t2}s)')
            self.balance = float(self.client.get_asset_balance(asset='FDUSD')['free'])
            self.q = '0.00000'
            print(f'{dt.today()} SELL_MARKET q:{q}')
            return self.sell_order
        except Exception as e:
            print(f'exception at _market_sell(): {e}')

    def run(self):
        self.ws.run_forever()

######################################################################################################################
######################################################################################################################
######################################################################################################################

class MakerBot:
    def __init__(self, symbol, itv, settings, API_KEY, SECRET_KEY, multi=25):
        self.cwd = getcwd()
        self.buy_slipp_file = '../settings/slippages_limit_buy.csv'
        self.sell_slipp_file = '../settings/slippages_limit_sell.csv'
        #sl_slipp_file = '/settings/slippages_StopLoss.csv'
        #with open(self.cwd+self.buy_slipp_file, 'a', newline='') as file: self.buy_slipp_wr = writer(file)
        #with open(self.cwd+self.sell_slipp_file, 'a', newline='') as file: self.sell_slipp_wr = writer(file)
        #with open(cwd+sl_slipp_file, 'a', newline='') as file: self.sl_slipp_wr = writer(file)

        self.symbol = symbol
        url = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{itv}'
        self.ws = WebSocketApp( url,
                                on_message=self.on_message,
                                on_error=self.on_error,
                                on_close=self.on_close,
                                on_open = self.on_open  )
        self.client = Client(API_KEY, SECRET_KEY)

        self.settings = settings
        prev_candles = self.client.get_historical_klines(symbol, itv, str(max(self.settings['MA_period'],self.settings['ATR_period'])*multi)+" minutes ago UTC")
        self.OHLCVX_data = deque([list(map(float,candle[1:6]+[0])) for candle in prev_candles[:-1]],
                                 maxlen=len(prev_candles[:-1]))
        self.init_balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        self.balance = self.init_balance
        self.signal = 0.0
        self.open_order, self.close_order, self.SL_order = None, None, None
        self.order_placed = 'null'
        self.req_p_buy, self.req_p_sell = 0,0
        self.retry_price = 0
        #print(self.OHLCVX_data)

    def on_message(self, ws, message):
        self.start_t = time()
        data = loads(message)
        #print(f"Received: {data}")
        data_k = data['k']
        if data_k['x']:
            self.OHLCVX_data.append(list( map(float, [data_k['o'],data_k['h'],data_k['l'],data_k['c'],data_k['v'],0]) ))
            self._analyze()
        close = float(data_k['c'])
        if self.order_placed=='buy':
            if self._check_order(self.open_order['orderId']):
                self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
                self._stop_loss(self.q, round(self.req_p_buy*(1-self.settings['SL']),2))
                self.order_placed = 'sl'
                with open(self.cwd+self.buy_slipp_file, 'a', newline='') as file:
                    writer(file).writerow([float(self.open_order['price'])/self.req_p_buy])
                self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
            else:
                print(f'RETRY BUY_LIMIT')
                adj_close = round(close-.01, 2)
                if adj_close>self.retry_price:
                    self.retry_price = adj_close
                    q = str(self.balance/adj_close)[:7]
                    self._cancel_all_orders()
                    self._limit_buy(q, adj_close)
        elif self.order_placed=='sell':
            if self._check_order(self.close_order['orderId']):
                self.order_placed = 'null'
                self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
                self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
                with open(self.cwd+self.sell_slipp_file, 'a', newline='') as file:
                    writer(file).writerow([float(self.close_order['price'])/self.req_p_sell])
            else:
                print(f'RETRY SELL_LIMIT')
                adj_close = round(close+.01, 2)
                if adj_close<self.retry_price:
                    self.retry_price = adj_close
                    self._cancel_all_orders()
                    self._limit_sell(self.q, adj_close)
        elif self.order_placed=='sl':
            if self._check_order(self.SL_order['orderId']):
                self.order_placed = 'null'
                self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
        print(f' INFO close:{close} signal:{self.signal:.3f} balance:{self.balance:.2f} self.q:{self.q} init_balance:{self.init_balance:.2f} order_placed:{self.order_placed}')

    def _analyze(self):
        self.signal, close = self._get_signal()
        if (self.signal >= self.settings['enter_at']) and (self.order_placed=='null') and self.balance>1.0:
            self.order_placed='buy'
            self.req_p_buy = close
            adj_close = round(close-.01, 2)
            self.retry_price = adj_close
            q = str(self.balance/adj_close)[:7]
            self._limit_buy(q, adj_close)
            #print(f' PLACED: {self.open_order}')
            print(f'(msg_to_exec_time: {time()-self.start_t}s)')
        elif (self.signal<=-self.settings['close_at']) and (self.order_placed=='null') and self.balance<1.0:
            self.order_placed='sell'
            self._cancel_all_orders()
            self.req_p_sell = close
            adj_close = round(close+.01, 2)
            self.retry_price = adj_close
            self._limit_sell(self.q, adj_close)
        #self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
        #self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]

    def _get_signal(self):
        OHLCV0_np = array(self.OHLCVX_data)
        ma = get_MA(array(self.OHLCVX_data), self.settings['typeMA'], self.settings['MA_period'])
        atr = ATR(OHLCV0_np[:,1], OHLCV0_np[:,2], OHLCV0_np[:,3], timeperiod=self.settings['ATR_period'])
        close = OHLCV0_np[-1,3]
        return (ma[-1]-close)/(atr[-1]*self.settings['ATR_multi'])

    def _check_order(self, ID):
        _order = self.client.get_order(symbol=self.symbol, orderId=ID)
        print(f' Checking order... {ID}')
        #print(f' Checking order: {_order}')
        if _order['status']=='FILLED':
            return True
        elif _order['status']=='PARTIALLY_FILLED':
            self.q = str(self.client.get_asset_balance(asset='BTC')['locked'])[:7]
            self.balance = float(self.client.get_asset_balance(asset='TUSD')['locked'])
        return False
    
    def _report_slipp(self):
        None

    '''def _get_filled_price(self, order):
        value, quantity = 0, 0
        for element in order['fills']:
            qty = element['qty']
            value += float(element['price'])*float(qty)
            quantity += qty
        return value/quantity'''
    
    def _cancel_all_orders(self):
        try:
            self.client._delete('openOrders', True, data={'symbol': self.symbol})
            print(f'DELETED ALL ORDERS')
        except Exception as e:  print(f'exception(_cancel_all_orders): {e}')
    
    def _limit_buy(self, q, price):
        print(f'BUY_LIMIT q:{q} price:{price}')
        try:
            self.open_order = self.client.order_limit_buy(  symbol=self.symbol,
                                                            quantity=q,
                                                            price=price )
            self.order_placed = 'buy'
        except Exception as e:  print(f'exception(_limit_buy): {e}')

    def _limit_sell(self, q, price):
        print(f'SELL_LIMIT q:{q} price:{price}')
        try:
            self.close_order = self.client.order_limit_sell( symbol=self.symbol,
                                                            quantity=q,
                                                            price=price )
            self.order_placed = 'sell'
        except Exception as e:  print(f'exception(_limit_sell): {e}')

    def _stop_loss(self, q, price):
        print(f'StopLoss_LIMIT q:{q} price:{price}')
        try:
            self.SL_order = self.client.create_order(symbol=self.symbol, 
                                                    side=SIDE_SELL,
                                                    type=ORDER_TYPE_STOP_LOSS_LIMIT, 
                                                    timeInForce=TIME_IN_FORCE_GTC, 
                                                    quantity=q, 
                                                    stopPrice=price, 
                                                    price=price  )
            #self.order_placed = 'sl'
        except Exception as e:  print(f'exception(_stop_loss): {e}')

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")
    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")
    def on_open(self, ws):
        print("### WebSocket opened ###")
        # For example, you can send a message immediately after it opens
        #ws.send("Hello there!")
    def run(self):
        self.ws.run_forever()