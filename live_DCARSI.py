from bots.DCARSI_binance import TakerBot
from credentials import binance_API_KEY, binance_SECRET_KEY

SYMBOL = 'BTCFDUSD'
buy_amount, enter_at, period = 5.00, 1.0, 14
INTERVAL = '1m'
# MA requires previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 2
SETTINGS = {'buy_amount': buy_amount,
            'enter_at': enter_at,
            'period': period}

if __name__ == '__main__':
    bot = TakerBot(SYMBOL,
                   INTERVAL,
                   SETTINGS,
                   binance_API_KEY,
                   binance_SECRET_KEY,
                   PREV_DATA_MULTIPLAYER)
    bot.run()
