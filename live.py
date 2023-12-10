from bots.bands_binance import TakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY


SYMBOL = 'BTCFDUSD'
MARKET = 'spot'
INTERVAL = '1m'
stop_loss, enter_at, close_at, typeMA, MA_period, ATR_period, ATR_multi = 0.0015, 0.5, 0.5, 1, 5, 5, 0.500
# MA requires previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 20
SETTINGS = {'stop_loss': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'ma_type': typeMA,
            'ma_period': MA_period,
            'atr_period': ATR_period,
            'atr_multi': ATR_multi}

if __name__ == '__main__':
    bot = TakerBot(SYMBOL,
                   MARKET,
                   INTERVAL,
                   SETTINGS,
                   binance_API_KEY,
                   binance_SECRET_KEY,
                   PREV_DATA_MULTIPLAYER)
    bot.run()