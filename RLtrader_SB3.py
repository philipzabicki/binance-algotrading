from datetime import datetime as dt

from stable_baselines3 import DQN

from definitions import TENSORBOARD_DIR, MODELS_DIR
from enviroments.rl_env import FuturesRL
from utils.get_data import by_BinanceVision
from utils.ta_tools import simple_rl_features_periods

# from time import sleep

# from sklearn.preprocessing import StandardScaler
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

TICKER = 'BTCUSDT'
ITV = '1m'
MARKET_TYPE = 'um'
DATA_TYPE = 'klines'
START_DATE = '2020-01-01'

if __name__ == "__main__":
    periods = [3, 5, 8, 13, 21, 34]
    # periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
    df = by_BinanceVision(ticker=TICKER,
                          interval=ITV,
                          market_type=MARKET_TYPE,
                          data_type=DATA_TYPE,
                          start_date=START_DATE,
                          split=False,
                          delay=345_600)
    print(f'df used: {df}')
    _, df_mark = by_BinanceVision(ticker=TICKER,
                                  interval=ITV,
                                  market_type=MARKET_TYPE,
                                  data_type='markPriceKlines',
                                  start_date=START_DATE,
                                  split=True,
                                  delay=345_600)
    print(f'df_mark used: {df_mark}')
    df = simple_rl_features_periods(df, periods, zscore_standardization=True).fillna(0.0)
    print(df)
    # plt.plot(df['Close'], label='Close')
    # plt.plot(df_mark['Close'], label='mark Close')
    # plt.legend(loc='upper left')
    # plt.show()

    # for col in df.columns:
    #     plt.plot(df[col], label=col)
    #     plt.legend(loc='upper left')
    #     plt.show()

    train_env = FuturesRL(df=df.iloc[:, 1:],
                          df_mark=df_mark,
                          dates_df=df['Opened'],
                          max_steps=1_440,
                          exclude_cols_left=5,
                          init_balance=1_000,
                          leverage=15,
                          fee=0.0005,
                          coin_step=0.001,
                          # slippage=get_market_slips_stats(),
                          render_range=120,
                          visualize=True, verbose=True, write_to_file=True)

    model = DQN("MlpPolicy",
                train_env,
                tensorboard_log=TENSORBOARD_DIR,
                verbose=2)

    model.learn(total_timesteps=750_000,
                progress_bar=True,
                log_interval=4)
    _date = str(dt.today()).replace(":", "-")[:-7]
    model_full_path = f'{MODELS_DIR}dqn_{_date}'
    model.save(model_full_path)

    del model
    model = DQN.load(model_full_path)

    test_env = FuturesRL(df=df.iloc[:, 1:],
                         df_mark=df_mark,
                         dates_df=df['Opened'],
                         max_steps=10_800,
                         exclude_cols_left=5,
                         init_balance=1_000,
                         leverage=33,
                         fee=0.0005,
                         coin_step=0.001,
                         # slippage=get_market_slips_stats(),
                         render_range=120,
                         visualize=True, verbose=True)
    obs = test_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = test_env.step(action)
        test_env.render()

    test_env.visualize = True
    obs = test_env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, info = test_env.step(action)
        test_env.render()
