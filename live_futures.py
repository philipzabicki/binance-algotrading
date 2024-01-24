from bots import FuturesTaker
from bots.signal_bot import MACDSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'USDT'
MARKET = 'um'
INTERVAL = '1m'
position_ratio, save_ratio, stop_loss, long_enter_at, long_close_at, short_enter_at, short_close_at = 0.5, 0.1, 0.01, 0.75, 0.75, 0.75, 0.75
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type, leverage = 12, 26, 9, 1, 1, 1, 54
trade_balance = 10.0
# MAs require previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 1
SETTINGS = {"trade_balance": trade_balance,
            "position_ratio": position_ratio,
            "save_ratio": save_ratio,
            "stop_loss": stop_loss,
            "long_enter_at": long_enter_at,
            "long_close_at": long_close_at,
            "short_enter_at": short_enter_at,
            "short_close_at": short_close_at,
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
