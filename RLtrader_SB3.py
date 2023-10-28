from TA_tools import simple_rl_features
from utility import seconds_since, get_market_slips_stats
from matplotlib import pyplot as plt
from get_data import by_BinanceVision
from enviroments.rl import SpotRL
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DQN

if __name__ == "__main__":
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600)
    df = simple_rl_features(df)
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
                         coin_step=0.0001,
                         slippage=get_market_slips_stats(),
                         render_range=120,
                         visualize=True)
    # print('Checking env...')
    # check_env(trading_env)

    model = DQN("MlpPolicy", trading_env,  learning_rate=0.001, verbose=2, device='cuda')
    model.learn(total_timesteps=432_000, log_interval=1, progress_bar=True)
    model.save("RLtrader")

    del model # remove to demonstrate saving and loading

    model = DQN.load("RLtrader")

    obs = trading_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = trading_env.step(action)
        trading_env.render()
        if done:
            obs = trading_env.reset()

    trading_env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action)
        if env.visualize: env.render()