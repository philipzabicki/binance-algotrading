from TA_tools import simple_rl_features, simple_rl_features_periods
from utility import seconds_since, get_market_slips_stats
from definitions import ROOT_DIR
# from matplotlib import pyplot as plt
from get_data import by_BinanceVision
from enviroments.rl import SpotRL
import torch as th
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600).tail(1_209_600)
    df = simple_rl_features_periods(df, [5, 8, 13, 21, 34, 55, 89, 144])
    '''for col in df.columns:
        print(col)
        plt.plot(df[col])
        plt.show()'''
    dates_df = df['Opened'].to_numpy()
    df = df.drop(columns='Opened').to_numpy()

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
    # vec_env = make_vec_env(trading_env, n_envs=4)
    # trading_env = make_vec_env(trading_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    arch = [215, 645, 323, 27]
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=arch)

    model = DQN("MlpPolicy",
                trading_env,
                # ent_coef=0.05,
                policy_kwargs=policy_kwargs,
                learning_starts=100_000,
                target_update_interval=1_000,
                tensorboard_log=ROOT_DIR + '/tensorboard/',
                verbose=2)
    '''
    model = DQN("MlpPolicy",
                policy_kwargs=policy_kwargs,
                learning_starts=10_800,
                env=trading_env,
                # learning_rate=0.001,
                # n_steps=100,
                # batch_size=128,
                # target_update_interval=1_000,
                verbose=2,
                tensorboard_log=ROOT_DIR+'/tensorboard/',
                device='cuda')
    '''
    model.learn(total_timesteps=10_000_000, log_interval=1, progress_bar=True)

    model.save("RLtrader")
    del model
    model = DQN.load("RLtrader")

    # Test without visualization for 3 days
    trading_env.max_steps = 604_800
    obs = trading_env.reset()[0]
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = trading_env.step(action)
        trading_env.render()

    trading_env.visualize = True
    obs = trading_env.reset()[0]
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = trading_env.step(action)
        trading_env.render()
