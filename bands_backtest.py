import time
import numpy as np 
from matplotlib import pyplot as plt
import mplfinance as mpf
import pandas as pd

from fileinput import filename
#import numpy as np
from enviroments.BacktestEnv import BacktestEnvSpot, BacktestEnv
from enviroments.BandParametrizerEnv import BandParametrizerEnv
import get_data
from statistics import mean
import TA_tools
from datetime import datetime
from dateutil.parser import parse

from TA_tools import add_MA_signal
from utility import minutes_since, get_slips_stats

if __name__=="__main__":
  SL,typeMA,MA_period,ATR_period,ATR_multi = 0.0043, 27, 165, 440, 1.267
  '''BTCTUSD_s = TA_tools.get_df(ticker='BTCTUSD', interval_list=['1m'], type='backtest', futures=False, indicator=None, period=None)
  BTCUSDT_f = TA_tools.get_df(ticker='BTCUSDT', interval_list=['1m'], type='backtest', futures=True, indicator=None, period=None)
  #df = pd.read_csv('C:/github/binance-trading/data/binance_data_spot/1s_data/BTCTUSD/BTCTUSD.csv').iloc[1_100_000:,:]
  #df['Opened'] = pd.to_datetime(df['Opened'], unit='ms')
  #print(df)
  dates_df = BTCTUSD_s['Opened'].to_numpy()
  BTCTUSD_s = BTCTUSD_s.drop(columns='Opened').to_numpy()
  BTCTUSD_s = np.hstack((BTCTUSD_s, BTCTUSD_s[:, -1:]))
  BTCTUSD_s = add_MA_signal(BTCTUSD_s,typeMA,MA_period,ATR_period,ATR_multi)'''
  df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=300)
  dates_df = df['Opened'].to_numpy()
  df = df.drop(columns='Opened').to_numpy()
  df = np.hstack((df, np.zeros((df.shape[0], 1))))
  df = add_MA_signal(df,typeMA,MA_period,ATR_period,ATR_multi)
  #=[ leverage=1, postition_ratio=1.000, typeMA=4, MA_period=11, ATR_period=408, ATR_multi=0.81 ]
  #14,16,334,0.3 46.38%
  #print('params: 10,28,296,0.19')
  #df = df_generator.add_particular_MAband(df, -1, 90, 90, 1.0)
  '''print(df[-100:,:])56,152,1.44
  dates_df = df['Opened']
  print(df[:500,:])
  pdf = pd.DataFrame(df[-1_000:,:-1], columns = ['Open','High','Low','Close'])
  pdf['Date'] = pd.to_datetime(dates_df.iloc[-1_000:,],  format='%Y-%m-%d %H:%M:%S')
  pdf.set_index('Date', inplace=True)
  apd = mpf.make_addplot(pd.DataFrame(df[:,-1]))
  mpf.plot(pdf, figratio=(8,4), type='candle', addplot=apd, volume=False, style='yahoo')'''

  '''df = df_generator.add_particular_MA(df, 16,10)
  print(df[-1])
  plt.title("Cóś") 
  plt.xlabel("Date") 
  plt.ylabel("4") 
  plt.plot(dates_df[-500:], df[-500:]) 
  plt.show()'''

  #time.sleep(1000)
  #dates_df = df['Opened'].to_numpy()
  #strat_env = BandParametrizerEnv(df=df[-minutes_since('22-03-2023'):,:], init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPPAGES, visualize=False, Render_range=60, write_to_csv=False)
  strat_env = BacktestEnvSpot(df=df[-minutes_since('23-03-2023'):,:], dates_df=dates_df[-minutes_since('23-03-2023'):],
                              init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=get_slips_stats(), StopLoss=SL,
                              Render_range=30, visualize=False, write_to_csv=False)

  strat_env.reset()
  done = False
  action = 0
  while not done:
    obs,reward,done,info = strat_env.step(action)
    if strat_env.qty == 0:
       if obs[-1]>=1: action = 1
       elif obs[-1]<=-1: action = 2
       else: action = 0
    elif strat_env.qty<0:
       if obs[-1]>=0: action = 1
       else: action = 0
    elif strat_env.qty>0:
       if obs[-1]<=0: action = 2
       else: action = 0
    if strat_env.visualize: strat_env.render()