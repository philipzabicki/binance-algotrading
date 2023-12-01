from bots.macd_binance import TakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY


SYMBOL = 'BTCFDUSD'
stop_loss, enter_at, close_at = 0.25, 1.0, 1.0
fast_ma_type, fast_period, slow_ma_type, slow_period, signal_ma_type, signal_period = 5, 6, 3, 23, 13, 12
INTERVAL = '1m'
# MA requires previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 150
SETTINGS = {'SL': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'fast_ma_type': fast_ma_type,
            'fast_period': fast_period,
            'slow_ma_type': slow_ma_type,
            'slow_period': slow_period,
            'signal_ma_type': signal_ma_type,
            'signal_period': signal_period,
            }

if __name__ == '__main__':
    bot = TakerBot(SYMBOL,
                   INTERVAL,
                   SETTINGS,
                   binance_API_KEY,
                   binance_SECRET_KEY,
                   PREV_DATA_MULTIPLAYER)
    bot.run()