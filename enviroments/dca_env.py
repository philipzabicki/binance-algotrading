from enviroments.base import BacktestEnv

class DCAEnv(BacktestEnv):
  def __init__(self, df, excluded_left=0, init_balance=28.5, position_ratio=1.0, leverage=1, fee=0.0, slippage=0.001, max_steps=0, lookback_window_size=1, Render_range=120, visualize=False):
     super().__init__(df, excluded_left, init_balance, position_ratio, leverage, fee, slippage, max_steps, lookback_window_size, Render_range, visualize)
     self.DCA_size = init_balance
     self.asset_balance = 0.0
     self.purchase_count = 0
     self.coin_step = .00001
  def _open_position(self, side, price):
    if side=='long':
      self.asset_balance += (self.DCA_size/price)
      self.purchase_count += 1
      self.episode_orders += 1
      #print(f'bought {self.DCA_size/price:.8f} new balance: {self.asset_balance}')
    elif side=='short':
      print('Shorting not allowed.')
  def _close_position(self, price, liquidated=False):
     print('You should never stop your DCA :)')
     return -1
  def _finish_episode(self):
    if self.df[self.current_step, 0]==self.df[-1, 0]:
      self.done = True
      print(f'Asset collected quantity: {self.asset_balance} purchase count: {self.purchase_count}')
      return self._next_observation(), None, self.done, {'asset_balance':self.asset_balance, 'purchase_count':self.purchase_count}
    else:
      None
  def step(self, action):
    super().step(action)
    info = {'asset_balance':self.asset_balance, 'purchase_count':self.purchase_count}
    return self._next_observation(), None, self.done, info