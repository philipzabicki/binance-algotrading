from websocket import WebSocketApp
from binance.client import Client
from json import loads
from numpy import array
from os import getcwd
from csv import writer
from collections import deque
from time import time
from TA_tools import get_MA
from talib import ATR
from binance.enums import *

class WebSocketClientBot:
    def __init__(self, symbol, itv, settings, API_KEY, SECRET_KEY, multi=25):
        cwd = getcwd()
        buy_slipp_file = '/settings/slippages_limit_buy.csv'
        sell_slipp_file = '/settings/slippages_limit_sell.csv'
        #sl_slipp_file = '/settings/slippages_StopLoss.csv'
        with open(cwd+buy_slipp_file, 'a', newline='') as file: self.buy_slipp_wr = writer(file)
        with open(cwd+sell_slipp_file, 'a', newline='') as file: self.sell_slipp_wr = writer(file)
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
        prev_candles = self.client.get_historical_klines(symbol, itv, str(max(MA_period,ATR_period)*multi)+" minutes ago UTC")
        self.OHLCVX_data = deque([list(map(float,candle[1:6]+[0])) for candle in prev_candles[:-1]],
                                 maxlen=len(prev_candles[:-1]))
        self.init_balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
        self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
        self.balance = self.init_balance
        self.signal = 0.0
        self.open_order, self.close_order, self.SL_order = None, None, None
        self.order_placed = 'null'
        self.req_p_buy, self.req_p_sell = 0,0
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
                self.order_placed = 'null'
                self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
                self._stop_loss(self.q, round(close*(1-self.settings['SL']),2))
                self.buy_slipp_wr.writerow([float(self.open_order['price'])/self.req_p_buy])
                self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
            else:
                print(f'RETRY BUY_LIMIT')
                self._cancel_all_orders()
                adj_close = round(close-.01, 2)
                q = str(self.balance/adj_close)[:7]
                self._limit_buy(q, adj_close)
        elif self.order_placed=='sell':
            if self._check_order(self.close_order['orderId']):
                self.order_placed = 'null'
                self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
                self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]
                self.sell_slipp_wr.writerow([float(self.close_order['price'])/self.req_p_sell])
            else:
                print(f'RETRY SELL_LIMIT')
                self._cancel_all_orders()
                adj_close = round(close+.01, 2)
                self._limit_sell(self.q, adj_close)
        print(f' INFO close:{close} signal:{self.signal:.3f} balance:{self.balance:.2f} self.q:{self.q} init_balance:{self.init_balance:.2f} order_placed:{self.order_placed}')

    def _analyze(self):
        self.signal, close = self._get_signal()
        if (self.signal >= self.settings['enter_at']) and (self.balance>1.0):
            self.order_placed='buy'
            self.req_p_buy = close
            adj_close = round(close-.01, 2)
            q = str(self.balance/adj_close)[:7]
            self._limit_buy(q, adj_close)
            #print(f' PLACED: {self.open_order}')
            print(f'(msg_to_exec_time: {time()-self.start_t}s)')
        elif (self.signal<=-close_at) and (self.balance<1.0):
            self.order_placed='sell'
            self._cancel_all_orders()
            self.req_p_sell = close
            adj_close = round(close+.01, 2)
            self._limit_sell(self.q, adj_close)
        #self.balance = float(self.client.get_asset_balance(asset='TUSD')['free'])
        #self.q = str(self.client.get_asset_balance(asset='BTC')['free'])[:7]

    def _get_signal(self):
        OHLCV0_np = array(self.OHLCVX_data)
        ma = get_MA(array(self.OHLCVX_data), self.settings['typeMA'], self.settings['MA_period'])
        atr = ATR(OHLCV0_np[:,1], OHLCV0_np[:,2], OHLCV0_np[:,3], timeperiod=self.settings['ATR_period'])
        close = OHLCV0_np[-1,3]
        return (ma[-1]-close)/(atr[-1]*self.settings['ATR_multi']), close

    def _check_order(self, ID):
        _order = self.client.get_order(symbol=self.symbol, orderId=ID)
        print(f' Checking order... {ID}')
        #print(f' Checking order: {_order}')
        if _order['status']=='FILLED':
            return True
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