from bands_bot import WebSocketClientBot
from credentials import binance_API_KEY,binance_SECRET_KEY

stop_loss,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi = 0.0015,0.5,0.5,1,5,5,0.500
SYMBOL = 'BTCTUSD'
INTERVAL = '1m'
BACK_DATA_MULTIPLER = 25
SETTINGS = {'SL': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'typeMA': typeMA,
            'MA_period': MA_period,
            'ATR_period': ATR_period,
            'ATR_multi': ATR_multi}

if __name__=='__main__':
    socket_url = f'wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{INTERVAL}'
    bot = WebSocketClientBot(   SYMBOL,
                                INTERVAL,
                                SETTINGS,
                                binance_API_KEY,
                                binance_SECRET_KEY,
                                BACK_DATA_MULTIPLER )
    bot.run()