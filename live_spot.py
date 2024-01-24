from bots import SpotTaker, MACDSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'FDUSD'
MARKET = 'spot'
INTERVAL = '1m'
stop_loss, enter_at, close_at = 0.013184332141448296, 0.5151211725660106, 0.013831228965725608
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type = 760, 677, 7, 36, 11, 4
# MAs require previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 1
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
    bot = MACDSignalsBot(bot_type=SpotTaker,
                         base=BASE_TICKER,
                         quote=QUOTE_TICKER,
                         market=MARKET,
                         itv=INTERVAL,
                         settings=SETTINGS,
                         API_KEY=binance_API_KEY,
                         SECRET_KEY=binance_SECRET_KEY,
                         multi=PREV_DATA_MULTIPLAYER)
    while True:
        bot.run()
