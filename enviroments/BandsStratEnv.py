from gc import collect
import numpy as np
from gym import spaces, Env
from enviroments.BacktestEnv import BacktestEnv, BacktestEnvSpot
from TA_tools import add_MA_signal

class OneRunEnvSpot(BacktestEnvSpot):
  def __init__(self, df, dates_df=None, excluded_left=0, StopLoss=0.0, enter_at=1.000, close_at=-1.000, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1,
               init_balance=100_000, postition_ratio=1.0, fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
               max_steps=0, Render_range=120, visualize=False, write_to_csv=False):
  #def __init__(*args, **kwargs):
     super().__init__(df=df, dates_df=dates_df, excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                      leverage=1, StopLoss=StopLoss, fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                      Render_range=Render_range, visualize=visualize, write_to_csv=write_to_csv)
     #super().__init__(*args, **kwargs)
     self.enter_threshold = enter_at
     self.close_threshold = close_at
     self.typeMA = typeMA
     self.MA_period = MA_period
     self.ATR_period = ATR_period
     self.ATR_multi = ATR_multi
  def reset(self, postition_ratio=1.0, StopLoss=0.01, enter_at=1.000, close_at=-1.000, typeMA=0, MA_period=1, ATR_period=1, ATR_multi=1.000):
     self.stop_loss = StopLoss
     self.enter_threshold = enter_at
     self.close_threshold = close_at
     self.postition_ratio = postition_ratio
     self.init_postition_size = self.init_balance*postition_ratio
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
     return obs, reward, done, info
  def _execute(self):
      # The first action must be initiated, I used 0
      # obs[-1]  asummes that indicator vlaues are stored in last column
      action = 0
      while not self.done:
        obs,reward,done,info = self.step(action)
        self.render()
        #print(f'SIGNAL: {obs[-1]:.2f}')
        if self.qty==0:
         if obs[-1]>=self.enter_threshold:
            action=1
         elif obs[-1]<=-self.enter_threshold:
            action=2
         else:
            action=0
        elif self.qty<0:
         if obs[-1]>=-self.close_threshold:
            action=1
         else:
            action=0
        elif self.qty>0:
         if obs[-1]<=self.close_threshold:
            action=2
         else:
            action=0
      #print(f'postition_ratio={self.postition_ratio}, leverage={self.leverage}, StopLoss={self.stop_loss:.4f}, enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}', end='')
      #print(f' typeMA={self.typeMA}, MA_period={self.MA_period}, ATR_period={self.ATR_period}, ATR_multi={self.ATR_multi:.3f}', end='')
      #print(f' reward={self.reward:.2f} (exec_time={info["exec_time"]:.2f}s)')
      return obs,reward,done,info

class BandsStratEnvSpot(Env):
    def __init__(self, df, dates_df=None, excluded_left=0, init_balance=100_000, postition_ratio=1.0,
                 fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
                 max_steps=0, Render_range=120, visualize=False, write_to_csv=False):
        self.exec_env = OneRunEnvSpot(df, dates_df=dates_df, StopLoss=0.01, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1.000,
                                      excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                                      fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                                      Render_range=Render_range, visualize=visualize, write_to_csv=write_to_csv)
        obs_lower_bounds = np.array([-np.inf for _ in range(8)])
        obs_upper_bounds = np.array([np.inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 1.000]
        action_upper = [0.0150, 1.000, 1.000, 32, 450, 500, 9.000]
        self.action_space = spaces.Box(low=np.array(action_lower), high=np.array(action_upper), dtype=np.float64)
    def reset(self, postition_ratio=1.0, StopLoss=0.01, enter_at=1.000, close_at=-1.000, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1.000):
        return self.exec_env.reset(postition_ratio, StopLoss, enter_at, close_at, typeMA, MA_period, ATR_period, ATR_multi)
    def step(self, action):
      self.reset(postition_ratio=1.0, StopLoss=round(action[0],4), enter_at=round(action[1],3), close_at=-round(action[2],3),
                 typeMA=int(action[3]), MA_period=int(action[4]), ATR_period=int(action[5]), ATR_multi=round(action[6],3))
      return self.exec_env._execute()
    
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