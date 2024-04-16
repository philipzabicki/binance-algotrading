from bots import FuturesTaker
from bots.signal_bot import MACDSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'USDT'
MARKET = 'um'
INTERVAL = '5m'
trade_balance = 21.5

# MACD settings:
position_ratio, save_ratio, stop_loss, take_profit, long_enter_at, long_close_at, short_enter_at, short_close_at = 0.9084360721923264, 0.0006522779237663826, 0.013771717681075216, 0.008168106994787865, 0.7325517706171805, 0.08116727153870686, 0.3755629310628192, 0.39153884632535846
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type, leverage = 808, 949, 427, 21, 31, 7, 19
SETTINGS = {"trade_balance": trade_balance,
            "position_ratio": position_ratio,
            "save_ratio": save_ratio,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
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

# Stoch settings:
# position_ratio, save_ratio, stop_loss, take_profit, long_enter_at, long_close_at, short_enter_at, short_close_at = 0.6065476973780412,0.9175248886714115,0.01045884148878456,0.017092000126018532,0.04696487050922178,0.04037982558669345,0.22028872498933774
# oversold_threshold, overbought_threshold, fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type, leverage = 22.63033474620837,50.34534916708502,209,6,584,15,7,45
# SETTINGS = {"trade_balance": trade_balance,
#             "position_ratio": position_ratio,
#             "save_ratio": save_ratio,
#             "stop_loss": stop_loss,
#             "take_profit": take_profit,
#             "long_enter_at": long_enter_at,
#             "long_close_at": long_close_at,
#             "short_enter_at": short_enter_at,
#             "short_close_at": short_close_at,
#             'oversold_threshold': oversold_threshold,
#             'overbought_threshold': overbought_threshold,
#             'fastK_period': fastK_period,
#             'slowK_period': slowK_period,
#             'slowD_period': slowD_period,
#             'slowK_ma_type': slowK_ma_type,
#             'slowD_ma_type': slowD_ma_type,
#             "leverage": leverage}

if __name__ == "__main__":
    bocik = MACDSignalsBot(bot_type=FuturesTaker,
                           base=BASE_TICKER,
                           quote=QUOTE_TICKER,
                           market=MARKET,
                           itv=INTERVAL,
                           settings=SETTINGS,
                           API_KEY=binance_API_KEY,
                           SECRET_KEY=binance_SECRET_KEY,
                           multi=1)
    while True:
        bocik.run()
