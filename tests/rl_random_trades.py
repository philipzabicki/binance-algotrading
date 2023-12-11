from enviroments.rl_env import SpotRL
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

if __name__ == "__main__":
    dates_df, df = by_BinanceVision(ticker='BTCFDUSD',
                                    interval='1s',
                                    market_type='spot',
                                    data_type='klines',
                                    start_date='2023-09-11',
                                    split=True,
                                    delay=0)

    trading_env = SpotRL(df=df,
                         dates_df=dates_df,
                         max_steps=86_400,
                         exclude_cols_left=4,
                         init_balance=1_000,
                         fee=0.0,
                         coin_step=0.001,
                         slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                         render_range=120,
                         visualize=True)

    obs = trading_env.reset()[0]
    terminated = False
    while not terminated:
        action = trading_env.action_space.sample()
        obs, reward, terminated, truncated, info = trading_env.step(action)
        trading_env.render()
