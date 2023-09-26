import numpy as np 
from matplotlib import pyplot as plt
#from enviroments.BacktestEnv import BacktestEnv, BacktestEnv
from enviroments.BandsStratEnv import BandsStratEnv
import get_data
from TA_tools import get_MA_signal
from utility import minutes_since, seconds_since, get_limit_slips_stats, get_market_slips_stats

if __name__=="__main__":
  SL,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi = 0.0001, 0.181, 0.45, 23, 2, 492, 3.894

  #df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=3_000)
  df = get_data.by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=172800)
  dates_df = df['Opened'].to_numpy()
  df = df.drop(columns='Opened').to_numpy()[-seconds_since('09-01-2023'):,:]
  df = np.hstack((df, np.zeros((df.shape[0], 1))))

  signal = get_MA_signal(df,typeMA,MA_period,ATR_period,ATR_multi)[:,-1]
  signal = signal[~np.isnan(signal)]
  print(signal)
  print(f'signal, mean:{np.mean(signal)} range:{np.ptp(signal)} std:{np.std(signal)}')
  plt.plot(signal)
  plt.show()
  # df=df[-minutes_since('23-03-2023'):,:].copy()
  strat_env = BandsStratEnv(df=df.copy(), dates_df=dates_df,
                                init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=get_market_slips_stats(),
                                Render_range=120, visualize=False)
  _,_,_,_ = strat_env.step([SL,enter_at,close_at,typeMA,MA_period,ATR_period,ATR_multi])
  plt.plot(strat_env.exec_env.PNLarrs[:,0])
  plt.show()
  plt.plot(strat_env.exec_env.PNLarrs[:,1])
  plt.show()
