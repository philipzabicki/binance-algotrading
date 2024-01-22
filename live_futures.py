from bots import FuturesTaker
from bots.signal_bot import MACDSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'USDT'
MARKET = 'um'
INTERVAL = '15m'
position_ratio, stop_loss, enter_at, close_at = 0.5027873928638713, 0.011830748094424587, 0.08935921017781712, 0.7720605257945228
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type, leverage = 77, 259, 74, 36, 33, 9, 54
# MAs require previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 25
SETTINGS = {"position_ratio": position_ratio,
            "stop_loss": stop_loss,
            "enter_at": enter_at,
            "close_at": close_at,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'fast_ma_type': fast_ma_type,
            'slow_ma_type': slow_ma_type,
            'signal_ma_type': signal_ma_type,
            "leverage": leverage}

if __name__ == "__main__":
    bocik = MACDSignalsBot(bot_type=FuturesTaker,
                           base=BASE_TICKER,
                           quote=QUOTE_TICKER,
                           market=MARKET,
                           itv=INTERVAL,
                           settings=SETTINGS,
                           API_KEY=binance_API_KEY,
                           SECRET_KEY=binance_SECRET_KEY,
                           multi=PREV_DATA_MULTIPLAYER)
    while True:
        bocik.run()
