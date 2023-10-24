from os import getcwd
import pandas as pd
from get_data import by_BinanceVision
from numpy import hstack, zeros

from utility import seconds_since, get_market_slips_stats
from enviroments.bands import BandsStratEnv
SLIPP = get_market_slips_stats()

data = pd.read_csv(getcwd()+'/reports/pop10242023-10-08 16-58-34.csv')
df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600)
df = df.drop(columns='Opened').to_numpy()
df = hstack((df, zeros((df.shape[0], 1))))
df = df[-seconds_since('09-01-2023'):,:]
env =  BandsStratEnv(df=df, init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPP)
for _,row in data.iterrows():
    action = [ row['StopLoss'], row['enter_at'], row['close_at'], row['typeMA'], row['MA_period'], row['ATR_period'], row['ATR_multi'] ]
    _, reward, _, info = env.step(action)
    if reward>0:
        print(f'{row}')