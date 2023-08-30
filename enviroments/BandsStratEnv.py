from gc import collect
from datetime import datetime as dt
import numpy as np
from gym import spaces, Env
from statistics import mean, stdev
from enviroments.BacktestEnv import BacktestEnv, BacktestEnvSpot
from TA_tools import add_MA_signal

class OneRunEnvSpot(BacktestEnvSpot):
  def __init__(self, df, dates_df=None, excluded_left=0, leverage=1, StopLoss=0.0, pos_enter_type=0, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1,
               init_balance=100_000, postition_ratio=1.0, fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
               max_steps=0, lookback_window_size=1, Render_range=120, visualize=False, write_to_csv=False):
  #def __init__(*args, **kwargs):
     super().__init__(df=df, dates_df=dates_df, excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                      leverage=leverage, StopLoss=StopLoss, fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                      lookback_window_size=lookback_window_size, Render_range=Render_range, visualize=visualize, write_to_csv=write_to_csv)
     #super().__init__(*args, **kwargs)
     self.pos_enter_type = pos_enter_type
     self.typeMA = typeMA
     self.MA_period = MA_period
     self.ATR_period = ATR_period
     self.ATR_multi = ATR_multi
  def reset(self, postition_ratio=1.0, leverage=1, StopLoss=0.01, pos_enter_type=0, typeMA=0, MA_period=1, ATR_period=1, ATR_multi=1.000):
     self.leverage = leverage
     self.stop_loss = StopLoss
     self.postition_ratio = postition_ratio
     self.init_postition_size = self.init_balance*postition_ratio
     self.pos_enter_type = pos_enter_type
     self.typeMA = typeMA
     self.MA_period = MA_period
     self.ATR_period = ATR_period
     self.ATR_multi = ATR_multi
     self.df = add_MA_signal(self.df, self.typeMA, self.MA_period, self.ATR_period, self.ATR_multi)
     super().reset()
     collect()
     return np.array([-1.0 for _ in range(7)], dtype="float32")
  def _finish_episode(self):
     _, reward, done, info = super()._finish_episode()
     obs =  np.array([self.episode_orders, self.cumulative_fees, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown, self.PL_count_mean, self.PL_ratio])
     #print(f'      action=[ leverage={self.leverage}, postition_ratio={self.postition_ratio:.3f}, typeMA={self.typeMA}, MA_period={self.MA_period}, ATR_period={self.ATR_period}, ATR_multi={self.ATR_multi:.2f} ]')
     return obs, reward, done, info
  def _execute(self):
      # The first action must be initiated, I used 0
      # obs[-1]  asummes that indicator vlaues are stored in last column
      action = 0
      while not self.done:
        obs,reward,done,info = self.step(action)
        #print(f'SIGNAL: {obs[-1]:.2f}')
        if self.pos_enter_type==0:
         if self.qty==0:
            if obs[-1]>=1: action=1
            elif obs[-1]<=-1: action=2
            else: action=0
         elif self.qty<0:
            if obs[-1]>=0: action=1
            else: action=0
         elif self.qty>0:
            if obs[-1]<=0: action=2
            else: action=0
        elif self.pos_enter_type==1:
         if self.qty==0:
            if obs[-1]>=1: action=1
            elif obs[-1]<=-1: action=2
            else: action=0
         elif self.qty<0:
            if obs[-1]>=1: action=1
            else: action=0
         elif self.qty>0:
            if obs[-1]<=-1: action=2
            else: action=0
        if self.visualize: self.render()
      #print(f'postition_ratio={self.postition_ratio}, leverage={self.leverage}, StopLoss={self.stop_loss:.4f},', end='')
      #print(f' typeMA={self.typeMA}, MA_period={self.MA_period}, ATR_period={self.ATR_period}, ATR_multi={self.ATR_multi:.3f}', end='')
      #print(f' reward={self.reward:.2f} (exec_time={info["exec_time"]:.2f}s)')
      return obs,reward,done,info
  
class OneRunEnv(BacktestEnv):
   def __init__(self, *args, **kwargs):
      super.__init__(*args, **kwargs)
      self.typeMA = kwargs['typeMA']
      self.MA_period = kwargs['MA_period']
      self.ATR_period = kwargs['ATR_period']
      self.ATR_multi = kwargs['ATR_multi']
   
class BandsStratEnv(Env):
    def __init__(self, *args, **kwargs):
      raise NotImplementedError

class BandsStratEnvSpot(Env):
    def __init__(self, df, df_mark=None, excluded_left=0, init_balance=100_000, postition_ratio=1.0,
                 fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
                 max_steps=0, lookback_window_size=1, Render_range=120, visualize=False, dates_df=None, write_to_csv=False):
        self.exec_env = OneRunEnvSpot(df, leverage=1, StopLoss=0.0, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1,
                                      excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                                      fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps, lookback_window_size=lookback_window_size,
                                      Render_range=Render_range, visualize=visualize, dates_df=dates_df, write_to_csv=write_to_csv)
        lower_bounds = np.array([-np.inf for _ in range(8)])
        upper_bounds = np.array([np.inf for _ in range(8)])
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds)
        #self.action_space = spaces.Box(low=np.array([0.0001, 0, 1, 1, 0.001]), high=np.array([0.0150, 33, 200, 500, 5.000]), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([0.0001, 0, 2, 1, 0.001]), high=np.array([0.0150, 32, 350, 500, 8.000]), dtype=np.float64)
        #self.action_space = spaces.Box(low=np.array([0.01, 1, 0.0001, 0, 2, 1, 0.01]), high=np.array([1.0, 125, 0.015, 32, 200, 500, 5]))
    def reset(self, postition_ratio=1.0, leverage=1, StopLoss=0.01, pos_enter_type=0, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1.250):
        #print(f'{self.exec_env.reset(postition_ratio, leverage, typeMA, MA_period, ATR_period, ATR_multi)}')
        #print(get_attributes_and_deep_sizes(self.exec_env))
        return self.exec_env.reset(postition_ratio, leverage, StopLoss, pos_enter_type, typeMA, MA_period, ATR_period, ATR_multi)
    def step(self, action):
      #print(f' ->(postition_ratio={round(action[0], 2):.2f}, leverage={int(action[1])}, StopLoss={round(action[2],4):.5f}, typeMA={int(action[3])}, MA_period={int(action[4])}, ATR_period={int(action[5])}, ATR_multi={action[6]:.2f})')
      #self.reset(postition_ratio=round(action[0], 2), leverage=int(action[1]), StopLoss=round(action[2],4), typeMA=int(action[3]), MA_period=int(action[4]), ATR_period=int(action[5]), ATR_multi=round(action[6], 3))
      #print(f' ->(postition_ratio={1}, leverage={1}, StopLoss={round(action[0],4):.5f}, typeMA={int(action[1])}, MA_period={int(action[2])}, ATR_period={int(action[3])}, ATR_multi={action[4]:.3f})', end=' ')
      #self.reset(postition_ratio=round(action[0], 2), leverage=int(action[1]), StopLoss=round(action[2],4), typeMA=int(action[3]), MA_period=int(action[4]), ATR_period=int(action[5]), ATR_multi=round(action[6], 3))
      self.reset(postition_ratio=1.0, leverage=1, StopLoss=round(action[0],4), typeMA=int(action[1]), MA_period=int(action[2]), ATR_period=int(action[3]), ATR_multi=round(action[4], 3))
      return self.exec_env._execute()