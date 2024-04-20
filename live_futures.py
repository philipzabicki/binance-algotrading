from bots import FuturesTaker
from bots.signal_bot import MACDSignalsBot
from credentials import binance_API_KEY, binance_SECRET_KEY

BASE_TICKER = 'BTC'
QUOTE_TICKER = 'USDT'
MARKET = 'um'
INTERVAL = '5m'
trade_balance = 5.0

# MACD settings:
position_ratio, save_ratio, stop_loss, take_profit, long_enter_at, long_close_at, short_enter_at, short_close_at = 0.9460166700548783, 0.0006627370950768449, 0.0071802375679447215, 0.0252682811737011, 0.9921070202773402, 0.27752799114024806, 0.793900445152847, 0.19245151212495437,
fast_period, slow_period, signal_period, fast_ma_type, slow_ma_type, signal_ma_type, leverage = 224, 291, 722, 35, 30, 5, 56
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
                           multi=1.1)
    while True:
        bocik.run()
