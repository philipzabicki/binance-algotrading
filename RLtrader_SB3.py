import TA_tools
from enviroments.RLEnv import RLEnvSpot
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DQN

if __name__=="__main__":
    df = TA_tools.get_df(ticker='BTCTUSD', interval_list=['1m'], type='trader', futures=False, indicator=None, period=None)
    dates_df = df['Opened'].to_numpy()
    df = df.drop(columns='Opened').to_numpy()
    df = TA_tools.add_particular_MAband(df,8,25,65,0.73)
    df = TA_tools.add_particular_MAband(df,0,5,61,0.57) # best RMA parameters
    df = TA_tools.add_particular_MAband(df,0,5,294,0.31) # best RMA parameters
    df = TA_tools.add_particular_MAband(df,1,9,1,1.12) # best SMA 
    df = TA_tools.add_particular_MAband(df,1,4,89,0.48) # best SMA
    df = TA_tools.add_particular_MAband(df,2,6,22,0.39) # best EMA
    df = TA_tools.add_particular_MAband(df,2,5,191,0.41) # best EMA
    df = TA_tools.add_particular_MAband(df,3,8,205,0.47) # best WMA
    df = TA_tools.add_particular_MAband(df,3,6,50,0.45) # best WMA
    env = RLEnvSpot(df=df[-58_111:,:], dates_df=dates_df[-58_111:], excluded_left=0, init_balance=600, postition_ratio=1.0, leverage=1, fee=0.0, slippage=0.0001, Render_range=120, visualize=True)
    #check_env(env)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000, log_interval=58_111)
    model.save("RLtrader")

    del model # remove to demonstrate saving and loading

    model = DQN.load("RLtrader")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs,reward,done,info = env.step(action)
        if env.visualize: env.render()