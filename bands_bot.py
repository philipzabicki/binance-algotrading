import websocket
from json import loads
from time import time
from csv import writer
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from statistics import mean, stdev
from collections import deque
from random import uniform
from talib import ATR
from TA_tools import get_MA
import os

from credentials import binance_API_KEY,binance_SECRET_KEY
client = Client(binance_API_KEY, binance_SECRET_KEY)

symbol = 'BTCTUSD'
interval = '1m'
SL,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi = 0.004, 0.744, 0.775, 27, 96, 44, 6.291

print(f'INITIAL PARAMETERS', SL,typeMA,MA_period,ATR_period,ATR_multi)
candles = client.get_historical_klines(symbol, interval, str(MA_period*300+100)+" minutes ago UTC")
#print(candles)
ohlc_v_data = deque([[float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5]), 0.0] for candle in candles[:-1]], maxlen=len(candles[:-1]))
#print(ohlc_v_data)
socket = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}'

SL_order, prevSL_order, market_sell, market_buy = None, None, None, None
init_balance = float(client.get_asset_balance(asset='TUSD')['free'])
est_daily_roi = 1
market_buy_slippages, market_sell_slippages, StopLoss_slippages = deque(maxlen=512), deque(maxlen=512), deque(maxlen=512)
prevSL_price, SL_price = 0, 0
buy_avg_slipp, buy_stdev_slipp, sell_avg_slipp, sell_stdev_slipp, StopLoss_avg_slipp, StopLoss_stdev_slipp = 0, 0, 0, 0, 0, 0
est_rois = {'D':deque(maxlen=360),'W':deque(maxlen=360),'M':deque(maxlen=360),'Q':deque(maxlen=360),'Y':deque(maxlen=360),'2Y':deque(maxlen=360)}
D,W,M,Q,Y,Y2 = 1,1,1,1,1,1
start_t, autoconfig_timer = time(), time()
def on_message(ws, message):
    global ohlc_v_data, balance, q, est_daily_roi, start_t, autoconfig_timer, symbol, typeMA, MA_period, ATR_period, ATR_multi, init_balance, SL, \
        prevSL_price, SL_price, SL_order, prevSL_order, market_sell, market_buy, \
        est_rois,D,W,M,Q,Y,Y2, market_buy_slippages, market_sell_slippages, StopLoss_slippages, StopLoss_avg_slipp, StopLoss_stdev_slipp, buy_avg_slipp, buy_stdev_slipp, sell_avg_slipp, sell_stdev_slipp
    data = loads(message)
    #print(data)
    data_k = data['k']
    _ohlc = np.array([float(data_k['o']), float(data_k['h']), float(data_k['l']), float(data_k['c']), float(data_k['v']), 0.0])
    _data = np.vstack([np.array(ohlc_v_data), _ohlc])
    #_data = np.hstack((_data, _data[:, -1:]))
    #print(f'_data {_data}')
    close = _data[-1,3]
    band_mid = get_MA(_data, typeMA, MA_period)
    #print(f'band_mid {type(band_mid)}')
    atr = ATR(_data[:,1], _data[:,2], _data[:,3], timeperiod=ATR_period)
    #print(f'atr {atr}')
    #watr = df_generator.VWATR(_data[:,1], _data[:,2], _data[:,3], _data[:,4], timeperiod=ATR_period)
    #signal = df_generator.anyMA_sig(close, band_mid[-1], atr[-1], atr_multi=ATR_multi)
    signal = (band_mid[-1]-close)/(atr[-1]*ATR_multi)

    balance = float(client.get_asset_balance(asset='TUSD')['free'])
    if data_k['x']:
        ohlc_v_data.append(_ohlc)

        adj_close = uniform(close, round(close*(1+0.00005), 2))
        print(f' INFO close:{close} band_mid:{round(band_mid[-1],2)} atr:{atr[-1]:.2f} signal:{signal:.3f} adj_close:{adj_close:.2f} balance:{balance:.2f} init_balance:{init_balance:.2f}')
        print(f'    EST_ROI(D/W/M/Q/Y):{D:.3f}/{W:.3f}/{M:.3f}/{Q:.3f}/{Y:.3f}/{Y2:.3f}')
        print(f'    current settings: {SL} {typeMA} {MA_period} {ATR_period} {ATR_multi}', )
        print(f'    market_buy_slippage(avg/stdev):{buy_avg_slipp:.7f}/{buy_stdev_slipp:.7f} ({len(market_buy_slippages)})')
        print(f'    market_sell_slippage(avg/stdev):{sell_avg_slipp:.7f}/{sell_stdev_slipp:.7f} ({len(market_sell_slippages)})')
        print(f'    SL_slippage(avg/stdev):{StopLoss_avg_slipp:.7f}/{StopLoss_stdev_slipp:.7f} ({len(StopLoss_slippages)})')
        if balance>1 and signal>=enter_at:
            q = str(balance/adj_close)[:7]
            market_buy = client.order_market_buy(symbol=symbol, quantity=q)
            filled_price = round(float(market_buy['cummulativeQuoteQty'])/float(market_buy['executedQty']), 2)
            prevSL_price = SL_price
            prevSL_order = SL_order
            SL_multipler, exc = 1.0, True
            while exc:
                try:
                    SL_price = round(filled_price*(1-SL*SL_multipler), 2)
                    SL_order = client.create_order( symbol=symbol, 
                                                    side=SIDE_SELL, 
                                                    type=ORDER_TYPE_STOP_LOSS_LIMIT, 
                                                    timeInForce=TIME_IN_FORCE_GTC, 
                                                    quantity=q, 
                                                    stopPrice=SL_price, 
                                                    price=SL_price )
                    exc = False
                except Exception as e:
                    print(f'exception(SL x{SL_multipler}): {e}')
                    SL_multipler *= 0.999
                    exc = True
            print(f'BUY_MARKET calc_q:{q} price: {close} filled_price: {filled_price}')
            print(f'STOP_LOSS LIMIT stopPrice:{SL_price} price:{SL_price} q:{q}')
            #print(SL_order)

            market_buy_slippages.append(filled_price/close)
            if len(market_buy_slippages)>1:
                buy_avg_slipp, buy_stdev_slipp = mean(market_buy_slippages), stdev(market_buy_slippages)
                csv_filename = os.getcwd()+'/settings/slippages_market_buy.csv'
                with open(csv_filename, 'a', newline='') as file:
                    _writer = writer(file)
                    #writer.writerow(header)
                    _writer.writerow([market_buy_slippages[-1]])
            last_25_orders = client.get_my_trades(symbol=symbol, limit=25)
            SL_value, SL_quantity = 0, 0
            for order in last_25_orders:
                if prevSL_price!=0 and prevSL_order!=None:
                    if order['orderId']==prevSL_order['orderId']:
                        SL_value += (float(order['qty'])*float(order['price']))
                        SL_quantity += float(order['qty'])
                        #print(order)
            if SL_value>0:
                StopLoss_slippages.append((SL_value/SL_quantity)/prevSL_price)
                csv_filename = os.getcwd()+'/settings/slippages_StopLoss.csv'
                with open(csv_filename, 'a', newline='') as file:
                    _writer = writer(file)
                    #writer.writerow(header)
                    _writer.writerow([StopLoss_slippages[-1]])
            if len(StopLoss_slippages)>1:
                StopLoss_avg_slipp, StopLoss_stdev_slipp = mean(StopLoss_slippages), stdev(StopLoss_slippages)
        elif balance<1 and signal<=-close_at:
            try:
                client._delete('openOrders', True, data={'symbol': symbol})
                print(f'DELETED STOP_LOSS LIMIT ORDER')
                '''order_id = client.get_open_orders(symbol=symbol)[-1]['orderId']
                print(f'order_id: {order_id}', end=' ')
                cancel = client.cancel_order(symbol=symbol, order_id=order_id)
                print(f'cancel: {cancel}')'''
            except Exception as e: print(f'exception(del SL order): {e}')
            market_sell = client.order_market_sell(symbol=symbol, quantity=q)
            filled_price = round(float(market_sell['cummulativeQuoteQty'])/float(market_sell['executedQty']), 2)
            market_sell_slippages.append(filled_price/close)
            if len(market_sell_slippages)>1:
                sell_avg_slipp, sell_stdev_slipp = mean(market_sell_slippages), stdev(market_sell_slippages)
                csv_filename = os.getcwd()+'/settings/slippages_market_sell.csv'
                with open(csv_filename, 'a', newline='') as file:
                    _writer = writer(file)
                    #writer.writerow(header)
                    _writer.writerow([market_sell_slippages[-1]])
            print(f'SELL_MARKET q: {q} price: {close} filled_price: {filled_price}')
        elif balance>1:
            time_diff = time()-start_t
            est_daily_roi = ((((balance/init_balance)-1)/time_diff)*86_400)+1
            est_rois['D'].append(est_daily_roi)
            est_rois['W'].append(est_daily_roi**7)
            est_rois['M'].append(est_daily_roi**30)
            est_rois['Q'].append(est_daily_roi**90)
            est_rois['Y'].append(est_daily_roi**365)
            est_rois['2Y'].append(est_daily_roi**730)
            D,W,M,Q,Y,Y2 = mean(est_rois['D']),mean(est_rois['W']),mean(est_rois['M']),mean(est_rois['Q']),mean(est_rois['Y']),mean(est_rois['2Y'])
            if time()-autoconfig_timer>=600:
                autoconfig_timer = time()
                print('#########################')
                print('UPDATING STRATEGY SETTINGS')
                df = pd.read_csv(os.getcwd()+'/reports/BTCTUSD1m_since0322_ATR.csv')
                print(df.iloc[-1,:])
                SL,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi = float(df.iloc[-1, -7]), df.iloc[-1, -6], df.iloc[-1, -5], int(df.iloc[-1, -4]), int(df.iloc[-1, -3]), int(df.iloc[-1, -2]), float(df.iloc[-1, -1])
                print(SL,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi)
                candles = client.get_historical_klines(symbol, interval, str(MA_period*300+100)+" minutes ago UTC")
                ohlc_v_data = deque([[float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5]), 0.0] for candle in candles[:-1]], maxlen=len(candles[:-1]))
                #print(ohlc_v_data)
                print('#########################')
ws = websocket.WebSocketApp(socket, on_message=on_message)
ws.run_forever()