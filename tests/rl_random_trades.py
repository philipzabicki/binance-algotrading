from TA_tools import simple_rl_features
from utility import seconds_since, get_market_slips_stats
from matplotlib import pyplot as plt
from get_data import by_BinanceVision
from enviroments.rl import SpotRL
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DQN

if __name__ == "__main__":
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600)
    #df = simple_rl_features(df)
    '''for col in df.columns:
        print(col)
        plt.plot(df[col])
        plt.show()'''
    dates_df = df['Opened'].to_numpy()[-seconds_since('09-01-2023'):]
    df = df.drop(columns='Opened').to_numpy()[-seconds_since('09-01-2023'):, :]

    trading_env = SpotRL(df=df,
                         dates_df=dates_df,
                         max_steps=86_400,
                         exclude_cols_left=4,
                         init_balance=1_000,
                         fee=0.0,
                         coin_step=0.001,
                         slippage=get_market_slips_stats(),
                         render_range=120,
                         visualize=True)

    obs = trading_env.reset()[0]
    terminated = False
    while not terminated:
        action = trading_env.action_space.sample()
        obs, reward, terminated, truncated, info = trading_env.step(action)
        trading_env.render()