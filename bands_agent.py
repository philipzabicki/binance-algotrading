from numpy import inf
from stable_baselines3 import DDPG

from enviroments import BandsStratSpotEnv
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

if __name__ == "__main__":
    slippages = get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market')
    # Parallel environments
    _, df = by_BinanceVision(ticker='BTCFDUSD',
                             interval='1m',
                             market_type='spot',
                             data_type='klines',
                             start_date='2023-09-11 00:00:00',
                             split=True,
                             delay=0)
    env = BandsStratSpotEnv(df=df,
                            # max_steps=259_200,
                            init_balance=300,
                            no_action_finish=inf,
                            fee=0.0,
                            coin_step=0.00001,
                            slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                            verbose=False)
    # check_env(env)
    # env = make_vec_env(env, n_envs=10)

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
    # env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_parametrizer")

    env.visualize = True
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    # model = A2C("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=25000)
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
