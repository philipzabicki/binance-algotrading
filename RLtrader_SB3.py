from TA_tools import simple_rl_features
from utility import seconds_since, get_market_slips_stats
from definitions import ROOT_DIR
# from matplotlib import pyplot as plt
from get_data import by_BinanceVision
from enviroments.rl import SpotRL
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
import torch as th

if __name__ == "__main__":
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600)
    df = simple_rl_features(df)
    '''for col in df.columns:
        print(col)
        plt.plot(df[col])
        plt.show()'''
    dates_df = df['Opened'].to_numpy()[-seconds_since('09-03-2023'):]
    df = df.drop(columns='Opened').to_numpy()[-seconds_since('09-03-2023'):, :]

    trading_env = SpotRL(df=df,
                         dates_df=dates_df,
                         max_steps=10_800,
                         exclude_cols_left=4,
                         init_balance=1_000,
                         fee=0.0,
                         coin_step=0.0001,
                         slippage=get_market_slips_stats(),
                         render_range=120,
                         visualize=False)
    # print('Checking env...')
    # check_env(trading_env)
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[2_187, 729, 243, 9])

    model = DQN("MlpPolicy",
                policy_kwargs=policy_kwargs,
                learning_starts=10_000,
                env=trading_env,
                # learning_rate=0.001,
                # n_steps=100,
                # batch_size=128,
                # target_update_interval=1_000,
                verbose=2,
                tensorboard_log=ROOT_DIR+'/tensorboard/',
                device='cuda')

    model.learn(total_timesteps=1080000, log_interval=1, progress_bar=True)
    model.save("RLtrader")
    del model

    model = DQN.load("RLtrader")
    trading_env.visualize = True
    obs = trading_env.reset()[0]
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = trading_env.step(action)
        trading_env.render()
