from matplotlib import pyplot as plt
from numpy import inf

from enviroments import BandsStratSpotEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_MA_band_signal
from utils.utility import get_slippage_stats

if __name__ == "__main__":
    ticker, interval, market_type, data_type, start_date = 'BTCFDUSD', '1m', 'spot', 'klines', '2023-09-11'
    action = [0.03568069167245792, 0.003006891123941202, 0.5101507449452755, 8, 723, 903, 11.126713604052402]

    dates_df, df = by_BinanceVision(ticker=ticker,
                                    interval=interval,
                                    market_type=market_type,
                                    data_type=data_type,
                                    start_date=start_date,
                                    split=True,
                                    delay=129_600)
    signals = get_MA_band_signal(df.to_numpy(), action[3], action[4], action[5], action[6])
    print(f'signal {signals}')
    plt.plot(signals[-1_000:])
    plt.axhline(y=action[1], color='green', linestyle='--')
    plt.axhline(y=-action[2], color='red', linestyle='--')
    plt.show()

    env = BandsStratSpotEnv(df=df, dates_df=dates_df, init_balance=300, no_action_finish=inf,
                            fee=0.0, coin_step=0.00001,
                            slippage=get_slippage_stats(market_type, ticker, interval, 'market'),
                            verbose=True, visualize=False)
    env.step(action)
