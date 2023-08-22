import time
import pandas as pd
from datetime import datetime as dt
from fileinput import filename
from enviroments.DCAEnv import DCAEnv
from statistics import mean
import TA_tools    

if __name__=="__main__":
  results = []
  for day in range(1,31):
    print(f'Day of month: {day}')
    for hour in range(24):
        print(f'hour of day: {hour}', end=' ')
        #[368:-218]
        df = TA_tools.get_df(ticker='XRPUSDT', interval_list=['1h'], type='backtest', futures=True, indicator=None, period=None)
        print(df.iloc[:,:5])
        df = df.iloc[661:-665,:]
        #print(df_filled)
        strat_env = DCAEnv(df=df.to_numpy(), excluded_left=0, init_balance=29, fee=0.0, slippage=0.0001, postition_ratio=0.00001, leverage=1, lookback_window_size=1, Render_range=50, visualize=False)
        obs, _, done, info = strat_env.reset()
        #print(obs)
        _counter = 0
        while not strat_env.done:
          #obs_current_date = dt.fromtimestamp(obs[0])
          if obs[0].hour==hour and obs[0].day==day:
            action = 1
            _counter += 1 
            #print(f'({obs[0]} should buy {_counter})')
            #time.sleep(0.1)
          else:
            action = 0
          obs, _, done, info = strat_env.step(action)
        results.append([day, hour, info['asset_balance'], info['purchase_count']])
        #time.sleep(1)
        #print(results)
  results = [el for el in results if el[0]<29]
  print(sorted(results, key=lambda item: item[2]))
          #print(obs)