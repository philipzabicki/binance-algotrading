from os import getcwd

import pandas as pd

from definitions import REPORT_DIR
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

SLIPP = get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market')

results = pd.read_csv(getcwd() + REPORT_DIR + 'pop10242023-10-08 16-58-34.csv')

_, df = by_BinanceVision(ticker='BTCFDUSD',
                         interval='1m',
                         market_type='spot',
                         data_type='klines',
                         start_date='2023-09-11 00:00:00',
                         split=True,
                         delay=0)

env = BandsStratSpotEnv(df=df, init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPP)

for _, row in results.iterrows():
    action = [row['StopLoss'], row['enter_at'], row['close_at'], row['typeMA'], row['MA_period'], row['ATR_period'],
              row['ATR_multi']]
    _, reward, _, _, _ = env.step(action)
    if reward > 0:
        print(f'{row}')
