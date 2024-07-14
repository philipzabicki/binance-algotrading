from numpy import inf

from enviroments.rl_env import SpotRL
from utils.get_data import by_BinanceVision

TICKER = 'BTCUSDT'
ITV = '1h'
MARKET_TYPE = 'spot'
DATA_TYPE = 'klines'
TRADE_START_DATE = '2023-12-04'
TRADE_END_DATE = '2024-06-01'
# Better to take more previous data for some TA features
DF_START_DATE = '2023-09-04'
DF_END_DATE = '2024-06-01'
ENV_KWARGS = {'max_steps': 2_160, # 90 days in 1h intervals
              # 'start_date': TRADE_START_DATE,
              # 'end_date': TRADE_END_DATE,
              'init_balance': 1_000,
              'no_action_finish': inf,
              'fee': .00075,
              'coin_step': 0.00001,
              'stop_loss': 0.075,
              'take_profit': 0.025,
              # 'slippage': get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              'verbose': True}


if __name__ == '__main__':
    df = by_BinanceVision(ticker=TICKER,
                          interval=ITV,
                          market_type=MARKET_TYPE,
                          data_type=DATA_TYPE,
                          # start_date=DF_START_DATE,
                          # end_date=DF_END_DATE,
                          split=False,
                          delay=0)
    print(f'df used: {df}')

    test_env = SpotRL(df=df, **ENV_KWARGS)
    print(test_env.reset())
    terminated = False
    while not terminated:
        action = test_env.action_space.sample()
        print(f'ACTION {action}')
        observation, reward, terminated, truncated, info = test_env.step(action)
        print(observation, reward, terminated, truncated, info)
