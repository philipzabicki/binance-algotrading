from bots import MACDSpotTakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'FDUSD'
MARKET = 'spot'
INTERVAL = '1m'
stop_loss, enter_at, close_at = 0.012128312331444244, 0.3468497701455136, 0.91450356736722
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type = 359, 877, 975, 26, 14, 13
# MAs require previous data longer than just calculation period size
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
    bot = MACDSpotTakerBot(base=BASE_TICKER,
                           quote=QUOTE_TICKER,
                           market=MARKET,
                           itv=INTERVAL,
                           settings=SETTINGS,
                           API_KEY=binance_API_KEY,
                           SECRET_KEY=binance_SECRET_KEY,
                           multi=PREV_DATA_MULTIPLAYER)
    bot.run()
