from bots.macd_binance import TakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY

SYMBOL = 'BTCFDUSD'
MARKET = 'spot'
INTERVAL = '1m'
stop_loss, enter_at, close_at = 0.011123635861121386, 0.32212581146447883, 0.5566056106449668
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type = 586, 810, 970, 9, 4, 23
# MA requires previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 25
SETTINGS = {'stop_loss': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'fast_ma_type': fast_ma_type,
            'slow_ma_type': slow_ma_type,
            'signal_ma_type': signal_ma_type}

if __name__ == '__main__':
    bot = TakerBot(SYMBOL,
                   MARKET,
                   INTERVAL,
                   SETTINGS,
                   binance_API_KEY,
                   binance_SECRET_KEY,
                   PREV_DATA_MULTIPLAYER)
    bot.run()
