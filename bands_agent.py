#import numpy as np
from enviroments.BandParametrizerEnv import BandParametrizerEnv
from TA_tools import get_df
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import ARS

if __name__=="__main__":
    slippages = {'market_buy':(1.000026, 0.000032), 'market_sell':(0.999971, 0.000029), 'SL':(1.0, 0.0)}
    # Parallel environments
    df = get_df(ticker='BTCTUSD', interval_list=['1m'], type='backtest', futures=False, indicator=None, period=None)
    dates_df = df['Opened'].to_numpy()
    df = df.drop(columns='Opened').to_numpy()
    env = BandParametrizerEnv(df=df[-136_304:,:], init_balance=1_000, fee=0.0, slippage=slippages, dates_df=False, visualize=False, Render_range=60, write_to_csv=False)
    #check_env(env)
    #env = make_vec_env(env, n_envs=10)

    '''from sb3_contrib import TQC

    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=10000, log_interval=4)'''
    '''model = RecurrentPPO("MlpLstmPolicy", env, verbose=2)
    model.learn(5000)

    env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
    print(mean_reward)

    model.save("ppo_recurrent")'''
    '''#n_actions = env.action_space.shape[-1]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = ARS("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("ars_pendulum")'''

    model = DDPG("MlpPolicy", env, verbose=1, device='cuda', learning_rate=0.001)
    model.learn(total_timesteps=10000, log_interval=5)
    model.save("ddpg_parametrizer")
    #env = model.get_env()

    del model # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_parametrizer")

    env.visualize=True
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    '''model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()      '''