from bots.bands_binance import TakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY


SYMBOL = 'BTCFDUSD'
stop_loss, enter_at, close_at, typeMA, MA_period, ATR_period, ATR_multi = 0.0015, 0.5, 0.5, 1, 5, 5, 0.500
INTERVAL = '1s'
# MA requires previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 20
SETTINGS = {'SL': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'typeMA': typeMA,
            'MA_period': MA_period,
            'ATR_period': ATR_period,
            'ATR_multi': ATR_multi}

if __name__ == '__main__':
    bot = TakerBot(SYMBOL,
                   INTERVAL,
                   SETTINGS,
                   binance_API_KEY,
                   binance_SECRET_KEY,
                   PREV_DATA_MULTIPLAYER)
    bot.run()