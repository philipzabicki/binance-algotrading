from bots import BandsSpotTakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'FDUSD'
MARKET = 'spot'
INTERVAL = '1m'
stop_loss, enter_at, close_at, ma_type, ma_period, atr_period, atr_multi = 0.0015, 0.5, 0.5, 1, 5, 5, 0.500
# MAs require previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 25
SETTINGS = {'stop_loss': stop_loss,
            'enter_at': enter_at,
            'close_at': close_at,
            'ma_type': ma_type,
            'ma_period': ma_period,
            'atr_period': atr_period,
            'atr_multi': atr_multi}

if __name__ == '__main__':
    bot = BandsSpotTakerBot(base=BASE_TICKER,
                            quote=QUOTE_TICKER,
                            market=MARKET,
                            itv=INTERVAL,
                            settings=SETTINGS,
                            API_KEY=binance_API_KEY,
                            SECRET_KEY=binance_SECRET_KEY,
                            multi=PREV_DATA_MULTIPLAYER)
    bot.run()
