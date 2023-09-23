from websocket import WebSocketApp
from json import loads, dumps
from collections import deque
from numpy import array 
from time import time
from binance.client import Client
from TA_tools import get_MA
from talib import ATR
from credentials import *
from mexc_sdk import Spot
ITV_ALIAS = {'Min1':'1m'}

class MEXCTakerBot:
    def __init__(self, symbol, interval, settings, API_KEYbinance, SECRET_KEYbinance,  API_KEYmexc, SECRET_KEYmexc, multi=25) -> None:
        self.symbol = symbol
        self.interval = interval
        self.settings = settings
        self.clientBinance = Client(API_KEYbinance, SECRET_KEYbinance)
        self.clientMEXC = Spot(API_KEYmexc, SECRET_KEYmexc)
        url = 'wss://wbs.mexc.com/ws'
        self.ws = WebSocketApp( url,
                                on_message=self.on_msg,
                                on_error=self.on_error,
                                on_close=self.on_close,
                                on_open=self.on_open    )
        prev_candles = self.clientBinance.get_historical_klines(symbol,
                                                         ITV_ALIAS[interval],
                                                         str(max(self.settings['MA_period'],self.settings['ATR_period'])*multi)+" minutes ago UTC")
        self.OHLCVX_data = deque([list(map(float,candle[1:6]+[0])) for candle in prev_candles[:-1]],
                                 maxlen=len(prev_candles[:-1]))
        self.msg_buffer = deque(maxlen=5)
        self._get_balances()
        self.init_balance = self.balance
        self.signal = 0.0

    def on_error(self, ws, error):
        print(f"Error occurred: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket closed ###")

    def on_open(self, ws):
        subscription_payload = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.kline.v3.api@{self.symbol}@{self.interval}"]
        }
        ws.send(dumps(subscription_payload))
        print("### WebSocket opened ###")

    def on_msg(self, ws, message):
        self.start_t = time()
        data = loads(message)
        self.msg_buffer.append(data)
        #print(data)
        ### If candle closed.
        if ((int(data['t'])//1_000) >= int(self.msg_buffer[-2]['d']['k']['T'])) and (self.msg_buffer[-2]['d']['k']['T']!=data['d']['k']['T']):
            data_dk = self.msg_buffer[-2]['d']['k']
            self.OHLCVX_data.append(list( map(float, [data_dk['o'],data_dk['h'],data_dk['l'],data_dk['c'],data_dk['v'],0]) ))
            print(f'CANDLE CLOSED {data_dk}')
            print(self.OHLCVX_data[-1])
            self._analyze()
        self._get_balances()
        print(f' INFO close:{data["d"]["k"]["c"]} signal:{self.signal:.3f} balance:{self.balance:.2f} self.q:{self.q}', end=' ')
        print(f'init_balance:{self.init_balance:.2f}')
    
    def _analyze(self):
        print('ANALYZING DATA')
        self.signal, close = self._get_signal()
        #if (self.signal >= self.settings['enter_at']) and (self.balance>1.0):
        if (self.signal >= self.settings['enter_at']):
            self.req_p_buy = close
            adj_close = round(close+.01, 2)
            q = str(self.balance/adj_close)[:7]
            self._market_buy(q)
            print(f'(delay(msg to buy order): {time()-self.start_t}s)')
            #self._report_slipp(self.buy_order, close, 'buy')
            self._stoploss(q, round(adj_close*(1-self.settings['SL']),2))
        #elif (self.signal<=-self.settings['close_at']) and (self.balance<1.0):
        elif (self.signal<=-self.settings['close_at']):
            self._cancel_all_orders()
            self._market_sell(self.q)
            print(f'(delay(msg to  sell order): {time()-self.start_t}s)')
            #self._report_slipp(self.sell_order, close, 'sell')
    
    def _get_signal(self):
        print('CHECKING SIGNAL')
        OHLCV0_np = array(self.OHLCVX_data)
        ma = get_MA(array(self.OHLCVX_data), self.settings['typeMA'], self.settings['MA_period'])
        atr = ATR(OHLCV0_np[:,1], OHLCV0_np[:,2], OHLCV0_np[:,3], timeperiod=self.settings['ATR_period'])
        close = OHLCV0_np[-1,3]
        return (ma[-1]-close)/(atr[-1]*self.settings['ATR_multi']), close

    def _get_balances(self):
        #print('UPDATING BALANCES')
        for element in self.clientMEXC.account_info()['balances']:
            if element['asset'] == 'USDT':
                self.balance = float(element['free'])
            if element['asset'] == 'XRP':
                self.q = element['free'][:8]

    def _cancel_all_orders(self):
        try:
            self.clientMEXC.cancel_open_orders(symbol=self.symbol)
            print(f'DELETED ALL ORDERS')
        except Exception as e:  print(f'exception(_cancel_all_orders): {e}')

    def _market_buy(self, q):
        print(f'BUY_MARKET q:{q}')
        self.buy_order = self.clientMEXC.new_order_test(self.symbol,
                                                   'BUY',
                                                   'MARKET',
                                                   options={"quantity":q})
        print(self.buy_order)

    def _market_sell(self, q):
        print(f'SELL_MARKET q:{q}')
        self.sell_order = self.clientMEXC.new_order_test(self.symbol,
                                                    'SELL',
                                                    'MARKET',
                                                    options={"quantity":q})
        print(self.sell_order)
    
    def _stoploss(self, q, price):
        print(f'StopLoss_LIMIT q:{q} price:{price}')
        self.stoploss_order = self.clientMEXC.new_order_test(self.symbol,
                                                    'SELL',
                                                    'STOP_LOSS_LIMIT',
                                                    options={"price":price, "quantity":q})
        print(self.stoploss_orders)

    def run(self):
        self.ws.run_forever()

stop_loss,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi = 0.0015,0.5,0.5,1,5,5,0.500
SETTINGS = {'SL': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'typeMA': typeMA,
            'MA_period': MA_period,
            'ATR_period': ATR_period,
            'ATR_multi': ATR_multi}

if __name__ == '__main__':
    bocik = MEXCTakerBot('BTCUSDT',
                         'Min1',
                         SETTINGS,
                         binance_API_KEY,
                         binance_SECRET_KEY,
                         MEXC_API_KEY,
                         MEXC_SECRET_KEY)
    bocik.run()