#from os import environ
#environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from numpy import array, float64, inf
from gym import spaces, Env
from enviroments.BacktestEnv import BacktestEnv, BacktestEnvFutures
from TA_tools import get_MA_signal
#from matplotlib import pyplot as plt
#import cProfile

class OneRunEnv(BacktestEnv):
  def __init__(self, df, dates_df=None, excluded_left=0, StopLoss=0.0, enter_at=1.000, close_at=-1.000, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1,
               init_balance=100_000, postition_ratio=1.0, fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
               max_steps=0, Render_range=120, visualize=False):
     #self.profiler = cProfile.Profile() 
     #self.profiler.enable()
     super().__init__(df=df, dates_df=dates_df, excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                      StopLoss=StopLoss, fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                      Render_range=Render_range, visualize=visualize)
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
    self.typeMA = typeMA
    self.MA_period = MA_period
    self.ATR_period = ATR_period
    self.ATR_multi = ATR_multi
    #print(f'OneRunEnv.reset {self.postition_ratio} {self.stop_loss} {self.enter_threshold} {self.close_threshold} {self.typeMA} {self.MA_period} {self.ATR_period} {self.ATR_multi}')
    super().reset()
    self.df[self.start_step:self.end_step,-1] = get_MA_signal(self.df[self.start_step:self.end_step,:], self.typeMA, self.MA_period, self.ATR_period, self.ATR_multi)
    self.obs = iter(self.df[self.start_step:self.end_step,:])
    return array([-1.0 for _ in range(7)], dtype="float32")
  def _finish_episode(self):
     #print('OneRunEnv._finish_episode()')
     super()._finish_episode()
     #obs =  array([self.episode_orders, self.cumulative_fees, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown, self.PL_count_mean, self.PL_ratio])
     if self.output:
         print(f' postition_ratio={self.postition_ratio}, StopLoss={self.stop_loss:.4f}, enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}', end='')
         print(f' typeMA={self.typeMA}, MA_period={self.MA_period}, ATR_period={self.ATR_period}, ATR_multi={self.ATR_multi:.3f}', end='')
         print(f' reward={self.reward:.2f} (exec_time={self.info["exec_time"]:.2f}s)')
     #self.profiler.disable()  # Zakończ profilowanie
     #self.profiler.print_stats(sort='cumtime')
     #return obs, reward, done, info
  def _execute(self):
      # The first action must be initiated, I used 0
      # obs[-1]  asummes that indicator vlaues are stored in last column
      action = 0
      while not self.done:
        obs,reward,done,info = self.step(action)
        #if self.visualize: self.render()
        #print(f'SIGNAL: {obs[-1]:.2f}')
        signal = obs[-1]
        action = 0
        if self.qty == 0:
           if signal >= self.enter_threshold:
              action = 1
           elif signal <= -self.enter_threshold:
              action = 2
        elif (self.qty<0) and (signal>=-self.close_threshold):
           action = 1
        elif (self.qty>0) and (signal<=self.close_threshold):
           action = 2
      #print(obs,reward,done,info)
      return obs,reward,done,info

class BandsStratEnv(Env):
    def __init__(self, df, dates_df=None, excluded_left=0, init_balance=100_000, postition_ratio=1.0,
                 fee=0.0002, coin_step=0.00001, slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)},
                 max_steps=0, Render_range=120, visualize=False):
        self.exec_env = OneRunEnv(df, dates_df=dates_df, StopLoss=0.01, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1.000,
                                  excluded_left=excluded_left, init_balance=init_balance, postition_ratio=postition_ratio,
                                  fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                                  Render_range=Render_range, visualize=visualize)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 0.001]
        action_upper = [0.0150, 1.000, 1.000, 31, 750, 750, 10.000]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)
    def reset(self, postition_ratio=1.0, StopLoss=0.01, enter_at=1.000, close_at=-1.000, typeMA=0, MA_period=2, ATR_period=2, ATR_multi=1.000):
        #print(f'BandsStratEnv.reset {postition_ratio} {StopLoss} {enter_at} {close_at} {typeMA} {MA_period} {ATR_period} {ATR_multi}')
        return self.exec_env.reset(postition_ratio, StopLoss, enter_at, close_at, typeMA, MA_period, ATR_period, ATR_multi)
    def step(self, action):
      #print(f'action: {action}')
      self.reset(   postition_ratio=1.0, StopLoss=round(action[0],4),
                    enter_at=round(action[1],3), close_at=-round(action[2],3),
                    typeMA=int(action[3]), MA_period=int(action[4]),
                    ATR_period=int(action[5]), ATR_multi=round(action[6],3)   )
      return self.exec_env._execute()
    
class OneRunEnvFutures(BacktestEnvFutures):
   def __init__(self, *args, **kwargs):
      super.__init__(*args, **kwargs)
      self.typeMA = kwargs['typeMA']
      self.MA_period = kwargs['MA_period']
      self.ATR_period = kwargs['ATR_period']
      self.ATR_multi = kwargs['ATR_multi']
   
class BandsStratEnvFutures(Env):
    def __init__(self, *args, **kwargs):
      raise NotImplementedError