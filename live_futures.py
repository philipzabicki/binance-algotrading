from bots import FuturesTaker
from bots.signal_bot import StochasticOscillatorSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'USDT'
MARKET = 'um'
INTERVAL = '1m'
position_ratio, save_ratio, stop_loss, long_enter_at, long_close_at, short_enter_at, short_close_at = 0.6065476973780412,0.9175248886714115,0.01045884148878456,0.017092000126018532,0.04696487050922178,0.04037982558669345,0.22028872498933774
oversold_threshold, overbought_threshold, fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type, leverage = 22.63033474620837,50.34534916708502,209,6,584,15,7,45
trade_balance = 50.0
# MAs require previous data longer than just calculation period size
PREV_DATA_MULTIPLAYER = 6
# MACD settings:
# SETTINGS = {"trade_balance": trade_balance,
#             "position_ratio": position_ratio,
#             "save_ratio": save_ratio,
#             "stop_loss": stop_loss,
#             "long_enter_at": long_enter_at,
#             "long_close_at": long_close_at,
#             "short_enter_at": short_enter_at,
#             "short_close_at": short_close_at,
#             'fast_period': fast_period,
#             'slow_period': slow_period,
#             'signal_period': signal_period,
#             'fast_ma_type': fast_ma_type,
#             'slow_ma_type': slow_ma_type,
#             'signal_ma_type': signal_ma_type,
#             "leverage": leverage}

# Stoch settings:
SETTINGS = {"trade_balance": trade_balance,
            "position_ratio": position_ratio,
            "save_ratio": save_ratio,
            "stop_loss": stop_loss,
            "long_enter_at": long_enter_at,
            "long_close_at": long_close_at,
            "short_enter_at": short_enter_at,
            "short_close_at": short_close_at,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold,
            'fastK_period': fastK_period,
            'slowK_period': slowK_period,
            'slowD_period': slowD_period,
            'slowK_ma_type': slowK_ma_type,
            'slowD_ma_type': slowD_ma_type,
            "leverage": leverage}

if __name__ == "__main__":
    bocik = StochasticOscillatorSignalsBot(bot_type=FuturesTaker,
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
