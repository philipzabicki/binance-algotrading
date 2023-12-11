from TA_tools import  simple_rl_features_periods # , signal_features_periods, simple_rl_features
# from utility import seconds_since, get_market_slips_stats
from definitions import ROOT_DIR
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from get_data import by_BinanceVision
from enviroments.rl_env import SpotRL
# from time import sleep
import torch as th
# from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN  # , PPO, A2C

# from sklearn.preprocessing import StandardScaler
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    # periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
    # 1_620_000 means 150 episodes 10_800 steps each (3 hours for 1s step)
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=604_800).tail(
        108_000 + max(periods) * 3)
    df = simple_rl_features_periods(df, periods, zscore_standardization=True)
    # df = signal_features_periods(df, periods)
    df.to_csv('C:/Users/philipz/Desktop/simple_rf_features.csv')

    '''
    for col in df.columns:
        print(df[col].describe())
        plt.figure()
        plt.plot(df[col])
        # Show/save figure as desired.
        plt.show()
        # print(f'column: {col}')
        # print(df[col].head(10))
        # plt.plot(df[col])
        # sleep(5)
        # plt.show()
    '''
    dates_df = df['Opened'].to_numpy()
    df = df.drop(columns='Opened').to_numpy()

    trading_env = SpotRL(df=df,
                         dates_df=dates_df,
                         max_steps=10_800,
                         exclude_cols_left=5,
                         init_balance=1_000,
                         fee=0.0,
                         coin_step=0.0001,
                         # slippage=get_market_slips_stats(),
                         render_range=120,
                         visualize=False)
    # print('Checking env...')
    # check_env(trading_env)
    # vec_env = make_vec_env(trading_env, n_envs=4)
    # trading_env = make_vec_env(trading_env, n_envs=8, vec_env_cls=SubprocVecEnv)

    arch = [140, 279, 279, 81]
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=arch)
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # gammas = [0.7]
    print(f'gammas tested {gammas}')
    for g in gammas:
        model = DQN("MlpPolicy",
                    trading_env,
                    gamma=g,
                    # batch_size=10,
                    # learning_starts=10_800,
                    # learning_rate=.00001,
                    # ent_coef=0.05,
                    # policy_kwargs=policy_kwargs,
                    # learning_starts=100_000,
                    # target_update_interval=3_000,
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
        model.learn(total_timesteps=10_800 * 15, log_interval=1, progress_bar=True)

    model.save("RLtrader")
    del model
    model = DQN.load("RLtrader")

    # Test without visualization for 3 days
    trading_env.max_steps = 10_800 * 250
    obs, _ = trading_env.reset()
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
