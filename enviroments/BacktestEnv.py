from gc import collect
import numpy as np
import pandas as pd
from  math import copysign, sqrt, floor
import time
from datetime import datetime as dt
from collections import deque
from random import randint
from gym import spaces, Env
from statistics import mean, stdev
from visualize import TradingGraph
from utility import linear_reg_slope, get_attributes_and_deep_sizes

class BacktestEnv(Env):
    def __init__(self, df, dates_df=None, df_mark=None, excluded_left=0, init_balance=1_000, postition_ratio=1.0, leverage=1, StopLoss=0.0, fee=0.0002, coin_step=0.001,
                 slippage={'market_buy':(1.0,0.0),'market_sell':(1.0,0.0),'SL':(1.0,0.0)}, max_steps=0, lookback_window_size=0, Render_range=120, visualize=False, write_to_csv=False):
        self.start_t = time.time()
        # https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
        self.POSITION_TIER = {  1:(125, .0040, 0), 2:(105, .005, 50), 
                                3:(50, .01, 1_300), 4:(20, .025, 46_300), 
                                5:(10, .05, 546_300), 6:(5, .10, 2_546_300), 
                                7:(4, .125, 5_046_300), 8:(3, .15, 8_046_300), 
                                9:(2, .25, 28_046_300), 10:(1, .50, 103_046_300)  }
        self.dates_df = dates_df
        try:
          print(f'start_date:{self.dates_df[0]}')
          print(f'end_date:{self.dates_df[-1]}')
        except: None
        self.df = df
        self.df_mark = df_mark
        self.exclude_count = excluded_left
        self.df_total_steps = len(self.df)-1
        
        self.coin_step = coin_step
        self.slippage = slippage
        self.fee = fee
        # BTCUSDTperp last 1Y mean=6.09e-05 stdev=6.52e-05, mean+2*stedv covers ~95,4% of variance
        #self.funding_rate = 0.01912 * (1/100)
        self.funding_rate = 0.01 * (1/100)
        self.max_steps = max_steps
        self.init_balance = init_balance
        self.init_postition_size = init_balance*postition_ratio
        self.postition_ratio = postition_ratio
        self.position_size = self.init_postition_size
        self.balance = self.init_balance
        self.leverage = leverage
        self.stop_loss = StopLoss
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range
        self.visualize = visualize
        self.write_to_csv = write_to_csv
        ###
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = spaces.Discrete(3)
        #self.action_space_n = len(self.action_space)
        ###
        # Observation space
        obs_space_dims = len(self.df[0, self.exclude_count:])
        lower_bounds = np.array([-np.inf for _ in range(obs_space_dims)])
        upper_bounds = np.array([np.inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds)
        #self.orders_history = deque(maxlen=self.lookback_window_size)
        #self.market_history = deque(maxlen=self.lookback_window_size)
        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, len(self.df[0])-self.exclude_count+6)

    # Reset the state of the environment to an initial state
    def reset(self):
        self.start_t = time.time()
        self.df_total_steps = len(self.df)-1
        self.done = False
        self.reward = 0
        if self.visualize: 
          self.trades = deque(maxlen=self.Render_range)
          self.visualization = TradingGraph(Render_range=self.Render_range)
        self.info = {}
        self.trades_PNL_ratio = []
        self.balance = self.init_balance
        self.position_size = self.init_postition_size
        self.margin = 0
        self.liquidation_price = 0
        self.prev_bal = 0
        self.tier = 0
        #self.stop_loss /= self.leverage
        self.stop_loss_price = 0
        self.qty = 0
        self.cumulative_fees = 0
        self.enter_price = 0
        self.liquidations = 0
        self.in_position = 0
        self.position_closed = 0
        self.in_position_counter = 0
        self.episode_orders = 0
        self.good_trades_count = 0
        self.profit_mean = 0
        self.loss_mean = 0
        self.bad_trades_count = 0
        self.max_drawdown = 0
        self.max_profit = 0
        self.SL_losses = 0
        self.loss_hold_counter = 0
        self.profit_hold_counter = 0
        self.realized_PNLs = []
        self.pnl = 0
        #self.cumulative_pnl = 0
        self.max_balance = self.balance
        self.min_balance = self.balance
        #self.balance_history = [self.balance, self.balance]
        #self.reward = 0

        if self.max_steps > 0:
            self.start_step = randint(self.lookback_window_size, self.df_total_steps - self.max_steps)
            self.end_step = self.start_step + self.max_steps
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
        self.current_step = self.start_step
        ### Garbage collector call ;)
        #print(get_attributes_and_deep_sizes(self))
        collect()
        return self._next_observation()

        '''for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            #self.orders_history.append([self.position_size, self.balance, self.in_position, copysign(1, self.qty), self.pnl, self.df[current_step, 4]-self.enter_price])
            self.market_history.append(self.df[self.current_step, self.exclude_count:])
        # backtest #
        return self.market_history.pop()
        #return np.concatenate((self.market_history, self.orders_history), axis=1)'''

    def _linear_slope_indicator(self, values: list):
      if len(values)<20:
        return -1
      else:
        _5 = len(values)//20
        percentile25 = linear_reg_slope(values[-_5*5:])
        #percentile50 = linear_reg_slope(values[-_5*10:])
        percentile75 = linear_reg_slope(values[-_5*15:])
        #percentile95 = linear_reg_slope(values[-_5*19:])
        slope_avg = percentile25-percentile75
        return copysign(abs(slope_avg)**(1/4), slope_avg)

    def _finish_episode(self):
      #print('_finish_episode')
      self.done = True
      #if (self.current_step==self.end_step) and self.good_trades_count>1 and self.bad_trades_count>1:
      if self.good_trades_count>1 and self.bad_trades_count>1:
        gain = self.balance-self.init_balance
        losses = [pnl if pnl<0 else 0 for pnl in self.realized_PNLs]
        profits = [pnl if pnl>0 else 0 for pnl in self.realized_PNLs]
        total_return_ratio = (self.balance/self.init_balance)-1
        mean_pnl = mean(self.realized_PNLs)
        stdev_pnl = stdev(self.realized_PNLs)
        self.profit_mean = mean(profits)
        self.loss_mean = mean(losses)
        if self.loss_hold_counter>0 and self.profit_hold_counter>0: hold_ratio = self.profit_hold_counter/self.loss_hold_counter
        else: hold_ratio = 1
        risk_free_return = (self.df[-1,3]/self.df[0,3])-1
        self.sharpe_ratio = (mean_pnl-risk_free_return)/stdev_pnl if stdev_pnl!=0 else -1
        self.sortino_ratio = (total_return_ratio-risk_free_return)/stdev(losses) if stdev(losses)!=0 else -1
        self.trades_ratio = self.good_trades_count/self.bad_trades_count
        self.pnl_means_ratio = abs(self.profit_mean/self.loss_mean)
        pnl_ratio = self.trades_ratio*self.pnl_means_ratio
        self.reward = copysign(gain*(1/(abs(stdev_pnl)**(1/7)))*sqrt(pnl_ratio)*(hold_ratio**(1/3))*self.episode_orders, gain)
        slope_indicator = self._linear_slope_indicator(self.trades_PNL_ratio)
        if self.reward<0 and slope_indicator<0:
          self.reward = self.reward*slope_indicator*-1
        else:
          self.reward = self.reward*slope_indicator
        if gain>0.25*self.init_balance:
        #if True:
          print(f'Episode finished: gain: ${gain:.2f} episode_orders: {self.episode_orders:_} cumulative_fees: ${self.cumulative_fees:.2f} liquidations: {self.liquidations} SL_losses: ${self.SL_losses:.2f}')
          print(f' trades_count(profit/loss): {self.good_trades_count:_}/{self.bad_trades_count:_}, trades_ROE(profit/loss): {self.profit_mean*100:.2f}%/{self.loss_mean*100:.2f}%, max(profit/drawdown): {self.max_profit*100:.2f}%/{self.max_drawdown*100:.2f}%')
          print(f' reward: {self.reward:.3f} sharpe_ratio: {self.sharpe_ratio:.2f} sortino_ratio: {self.sortino_ratio:.2f} hold_ratio: {hold_ratio:.2f} pnl_ratio: {pnl_ratio:.2f} stdev_pnl: {stdev_pnl:.5f} slope_avg: {slope_indicator:.4f}')
        self.info = {'gain':gain, 'pnl_ratio':pnl_ratio, 'stdev_pnl':stdev_pnl, 'position_hold_sums_ratio':hold_ratio, 'slope_indicator':slope_indicator, 'exec_time':time.time()-self.start_t}
      else:
        self.reward = -np.inf
        self.sharpe_ratio,self.sortino_ratio,self.trades_ratio,self.pnl_means_ratio = -1,-1,-1,-1
        self.info = {'gain':0, 'episode_orders':self.episode_orders, 'pnl_ratio':0, 'stdev_pnl':0, 'position_hold_sums_ratio':0, 'slope_indicator':0, 'exec_time':time.time()-self.start_t}
        #print(f'EPISODE FAILED! (end_step not reached OR profit/loss trades less than 2)')
      if self.write_to_csv:
        filename = str(self.__class__.__name__)+str(dt.today())[:-7].replace(':','-')+'.csv'
        pd.DataFrame(self.trades_PNL_ratio).to_csv(filename, index=False)
        print(f'writing to file: {filename}')
      return self._next_observation(), self.reward, self.done, self.info

    # Get the data points for the given current_step
    def _next_observation(self):
      return self.df[self.current_step, self.exclude_count:]

    def _get_pnl(self, price, update=False):
      #print('Starting _get_pnl')
      _pnl = (((price/self.enter_price)-1)*self.leverage)*copysign(1, self.qty)
      #print('Calculated _pnl')
      #print(f'_get_pnl {_pnl} price:{price} update:{update} self.enter_price:{self.enter_price} self.qty:{self.qty}')
      if update: 
          self.pnl = _pnl
          #print('Updated pnl')
          #self.pnl_list.append(self.pnl)
      elif not self.in_position:
          self.pnl = 0
          #print('Set pnl to 0 because not in position')
      #print('Ending _get_pnl')
      return _pnl

    def _check_tier(self):
      #print('_check_tier')
      if self.position_size<50_000: self.tier = 1
      elif 50_000<self.position_size<250_000: self.tier = 2
      elif 250_000<self.position_size<3_000_000: self.tier = 3
      elif 3_000_000<self.position_size<15_000_000: self.tier = 4
      elif 15_000_000<self.position_size<30_000_000: self.tier = 5
      elif 30_000_000<self.position_size<80_000_000: self.tier = 6
      elif 80_000_000<self.position_size<100_000_000: self.tier = 7
      elif 100_000_000<self.position_size<200_000_000: self.tier = 8
      elif 200_000_000<self.position_size<300_000_000: self.tier = 9
      elif 300_000_000<self.position_size<500_000_000: self.tier = 10
    
    '''def _check_margin(self):
      #print('_check_margin')
      if self.qty>0:
        min_price = self.df[self.current_step, 2]
      elif self.qty<0:
        min_price = self.df[self.current_step, 1]
      else:
        pass
        #print('co Ty tu robisz?')
      position_value = abs(self.qty*min_price)
      unrealized_PNL = abs(self.qty*self.enter_price/self.leverage)*self._get_pnl(min_price)
      # 1.25% Liquidation Clearance
      margin_balance = self.margin + unrealized_PNL - (position_value*0.0125) - (position_value*self.fee)
      maintenance_margin = position_value*self.POSITION_TIER[self.tier][1]-self.POSITION_TIER[self.tier][2]
      print(f'min_price:{min_price:.2f} position_value:{position_value:.2f} unrealized_PNL:{unrealized_PNL:.2f} Clearance:{(position_value*0.0125)} fee:{(position_value*self.fee)} margin:{self.margin} margin_balance:{margin_balance:.2f} maintenance_margin:{maintenance_margin:.2f} margin_ratio:{maintenance_margin/margin_balance*100}')
      if maintenance_margin>margin_balance:
        return True
      else:
        return False'''
    
    def _check_margin(self):
      #print('_check_margin')
      self.liquidation_price = (self.margin-self.qty*self.enter_price)/(abs(self.qty)*self.POSITION_TIER[self.tier][1]-self.qty)
      #print(f'liquidation_price:{self.liquidation_price} (margin:{self.margin} qty:{self.qty} enter_price:{self.enter_price})')
      if self.qty>0:
        min_price = self.df_mark[self.current_step, 2]
        if self.liquidation_price>=min_price: return True
      elif self.qty<0:
        max_price = self.df_mark[self.current_step, 1]
        if self.liquidation_price<=max_price: return True
      else:
        return False

    def _open_position(self, side, price):
      if side=='long':
        if self.visualize:
          self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total': self.qty, 'type': "open_long"})
          print(f'OPENING LONG at {price}')
        rnd_factor = np.random.normal(self.slippage['market_buy'][0], self.slippage['market_buy'][1], 1)[0]
        price = round(price*rnd_factor, 2)
        self.stop_loss_price = round((1-self.stop_loss)*price,2)
      elif side=='short':
        if self.visualize:
          self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total': self.qty, 'type': "open_short"})
          print(f'OPENING SHORT at {price}')
        rnd_factor = np.random.normal(self.slippage['market_sell'][0], self.slippage['market_sell'][1], 1)[0]
        price = round(price*rnd_factor, 2)
        self.stop_loss_price = round((1+self.stop_loss)*price,2)
      #print(f'OPENING x{self.leverage} {side} at price {price} ')
      self.in_position = 1
      self.episode_orders += 1
      self.enter_price = price
      self._check_tier()
      if self.leverage>self.POSITION_TIER[self.tier][0]:
        print(f' Leverage exceeds tier {self.tier} max, changing from {self.leverage} to {self.POSITION_TIER[self.tier][0]} (Balance: ${self.balance})')
        #print(f'Balance: {self.balance}')
        self.leverage = self.POSITION_TIER[self.tier][0]
      adj_qty = floor(self.position_size*self.leverage/(price*self.coin_step))
      if adj_qty==0:
        adj_qty=1
        #print('Forcing adj_qty to 1. Calculated quantity possible to buy with given postion_size and coin_step equals 0')
      self.margin = (adj_qty*self.coin_step*price)/self.leverage
      #print(f'  (position size {self.position_size} qty {adj_qty})')
      if self.margin>self.balance: return self._finish_episode()
      self.prev_bal = self.balance
      self.balance -= self.margin
      fee = (self.margin*self.fee*self.leverage)
      #print(f'  (fee {fee:.8f})')
      self.margin -= fee
      #print(f'OPENING POSITION fee:{fee:.2f} Margin:{self.margin:.2f} Balance:{self.balance+self.margin:.2f}')
      self.cumulative_fees += fee
      #print(f' balance(minus position size and fee) {self.balance}')
      if side=='long':
        self.qty = adj_qty*self.coin_step
      elif side=='short':
        self.qty = -1*adj_qty*self.coin_step
      #time.sleep(2)

    def _close_position(self, price, liquidated=False, SL=False):
      if SL:
        rnd_factor = np.random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
        if self.qty>0:
          while price*rnd_factor>self.enter_price:
            rnd_factor = np.random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
        elif self.qty<0:
          while price*rnd_factor<self.enter_price:
            rnd_factor = np.random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
      else:
        if self.qty>0:  rnd_factor = np.random.normal(self.slippage['market_sell'][0], self.slippage['market_sell'][1], 1)[0]
        elif self.qty<0:  rnd_factor = np.random.normal(self.slippage['market_buy'][0], self.slippage['market_buy'][1], 1)[0]
      if self.visualize:
        if self.qty>0:
          if SL: trade_type= "SL_long"
          elif liquidated: trade_type= "liquidate_long"
          else: trade_type= "close_long"
          print(f'CLOSING LONG at {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
        elif self.qty<0:
          if SL: trade_type="SL_short"
          elif liquidated: trade_type="liquidate_short"
          else: trade_type= "close_short"
          print(f'CLOSING SHORT at {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
        self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total':self.qty, 'type':trade_type})
      price = round(price*rnd_factor, 2)
      #print(f'CLOSING position at price {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
      _position_value = abs(self.qty)*price
      _fee = (_position_value*self.fee)
      if liquidated:
        margin_balance = 0
      else:
        unrealized_PNL = (abs(self.qty)*self.enter_price/self.leverage)*self.pnl
        margin_balance = self.margin+unrealized_PNL-_fee
      #print(f'CLOSING POSITION fee:{_fee:.2f} UPNL:{unrealized_PNL:.2f} MarginBalance:{margin_balance:.2f} Balance:{self.balance:.2f}')
      self.cumulative_fees += _fee
      self.balance += margin_balance
      self.margin = 0
      percentage_profit = (self.balance/self.prev_bal)-1
      self.realized_PNLs.append(percentage_profit)
      ### PROFIT
      if percentage_profit>0:
        if self.balance >= self.max_balance:
          self.max_balance = self.balance
          if self.balance>1_000_000:
            print(f'$$$$$$$$ {self.balance} $$$$$$$$')
        self.good_trades_count += 1
        if self.max_profit==0 or percentage_profit>self.max_profit:
          self.max_profit = percentage_profit
      ### LOSS
      elif percentage_profit<0:
        if self.balance <= self.min_balance: self.min_balance = self.balance
        self.bad_trades_count += 1
        if self.max_drawdown==0 or percentage_profit<self.max_drawdown:
          self.max_drawdown = percentage_profit
        if SL:
          self.SL_losses += (self.balance-self.prev_bal)
      if self.good_trades_count>0 and self.bad_trades_count>0:
        self.trades_PNL_ratio.append(self.good_trades_count/self.bad_trades_count)
      self.position_size = (self.balance*self.postition_ratio)
      self.qty = 0
      self.in_position = 0
      self.in_position_counter = 0
      self.stop_loss_price = 0
      self.pnl = 0
      #time.sleep(2)
    
    # Execute one time step within the environment
    def step(self, action):
        #print(f'current_step:{self.current_step} start_step:{self.start_step} end_step:{self.end_step} margin:{self.margin} balance:{self.balance}')
        close = self.df[self.current_step, 3]
        if self.current_step==self.end_step:
          if self.in_position: self._close_position(close)
          return self._finish_episode()
        if not self.in_position:
          if action == 0:
            if ((self.current_step-self.start_step)>10_080) and (not self.episode_orders):
              #print(f'(episode finished: {self.episode_orders} trades {self.current_step-self.start_step} steps)')
              return self._finish_episode()
          elif self.position_size<=((close*self.coin_step)/self.leverage)+((close*self.coin_step)*self.fee):
            return self._finish_episode()
          elif action == 1:
            self._open_position('long', close)
          elif action == 2:
            self._open_position('short', close)
        elif self.in_position:
          self.in_position_counter+=1
          if self.in_position_counter%480==0:
            self.margin -= (abs(self.qty)*close*self.funding_rate)
          self._get_pnl(close, update=True)
          if self.pnl<0: self.loss_hold_counter +=1
          elif self.pnl>0: self.profit_hold_counter +=1
          if (self.df[self.current_step, 2]<=self.stop_loss_price and self.qty>0) or (self.df[self.current_step, 1]>=self.stop_loss_price and self.qty<0):
            self._close_position(self.stop_loss_price, SL=True)
          elif self._check_margin():
            self.liquidations += 1
            self._close_position(close, liquidated=True)
          else:
            if action == 0:
              pass
            elif (action==1 and self.qty<0) or (action==2 and self.qty>0):
              self._close_position(close)
        info = {'action': action,
                'reward': 0,
                'step': self.current_step,
                'exec_time':time.time()-self.start_t}
        self.current_step += 1
        #time.sleep(0.1)
        return self._next_observation(), 0, self.done, info
        
    # Render environment
    def render(self, visualize=False, *args, **kwargs):
      if visualize or self.visualize:
        Date = self.dates_df[self.current_step]
        Open = self.df[self.current_step, 0]
        High = self.df[self.current_step, 1]
        Low = self.df[self.current_step, 2]
        Close = self.df[self.current_step, 3]
        #Volume = self.df[self.current_step, 4]
        Volume = self.df[self.current_step, -1]
        # Render the environment to the screen
        if self.in_position: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance+(self.position_size+(self.position_size*self.pnl)), self.trades)
        else: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)
      else:
        print('No dates df.')

######################################################################################################################
######################################################################################################################
######################################################################################################################
class BacktestEnvSpot(BacktestEnv):
  def step(self, action):
        # 30-day DCA, adding 222USD to balance
        #if (self.current_step-self.start_step)%43_200 == 0:
          #self.balance+=222
          #self.init_balance+=222
        close = self.df[self.current_step, 3]
        if self.current_step==self.end_step:
          if self.in_position:
            self._sell(close)
          return self._finish_episode()
        if not self.in_position:
          if action == 0:
            ## Finish episode if there was 0 orders after 7 days (24*60*7)
            if ((self.current_step-self.start_step)>10_080) and (not self.episode_orders):
              #print(f'(episode finished: {self.episode_orders} trades {self.current_step-self.start_step} steps)')
              return self._finish_episode()
            else:
              pass
          elif action == 1:
            if self.balance*(1-self.fee)<=(close*self.coin_step):
              return self._finish_episode()
            else:
              self._buy(close)
        elif self.in_position:
          self.in_position_counter+=1
          self._get_pnl(close, update=True)
          if self.pnl<0: self.loss_hold_counter +=1
          elif self.pnl>0: self.profit_hold_counter +=1
          if self.df[self.current_step, 2]<=self.stop_loss_price:
            self._sell(self.stop_loss_price, SL=True)
          elif action == 0:
              pass
          elif action==2 and self.qty>0:
              self._sell(close)
        info = {'action': action,
                'reward': 0,
                'step': self.current_step}
        self.current_step += 1
        return self._next_observation(), 0, self.done, info

  def _buy(self, price):
      if self.visualize: self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total': self.qty, 'type': "open_long"})
      market_buy_rnd_factor = np.random.normal(self.slippage['market_buy'][0], self.slippage['market_buy'][1], 1)[0]
      price = round(price*market_buy_rnd_factor, 2)
      self.in_position = 1
      self.episode_orders += 1
      self.enter_price = price
      self.stop_loss_price = round((1-self.stop_loss)*price,2)
      ### When there is no fee, substract 1 just to be sure balance can buy this amount
      step_adj_qty = floor((self.position_size*(1-3*self.fee))/(price*self.coin_step))
      if step_adj_qty==0: 
        self._finish_episode()
      self.qty = round(step_adj_qty*self.coin_step, 5)
      self.position_size = round(self.qty*price, 2)
      self.prev_bal = self.balance
      self.balance -= self.position_size
      fee = (self.position_size*self.fee)
      self.balance -= fee
      self.cumulative_fees += fee

  def _sell(self, price, SL=False):
      if SL:
        SL_rnd_factor = np.random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
        while price*SL_rnd_factor>self.enter_price:
          SL_rnd_factor = np.random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
        price = round(price*SL_rnd_factor, 2)
        if self.visualize: self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total': self.qty, 'type': "open_short"})
      else:
        market_sell_rnd_factor = np.random.normal(self.slippage['market_sell'][0], self.slippage['market_sell'][1], 1)[0]
        price = round(price*market_sell_rnd_factor, 2)
        if self.visualize: self.trades.append({'Date' : self.dates_df[self.current_step], 'High' : self.df[self.current_step, 1], 'Low' : self.df[self.current_step, 2], 'total': self.qty, 'type': "close_long"})
      self.balance += round(self.qty*price, 2)
      fee = abs(price*self.qty*self.fee)
      self.balance -= fee
      self.cumulative_fees += fee
      percentage_profit = (self.balance/self.prev_bal)-1
      self.realized_PNLs.append(percentage_profit)
      ### PROFIT
      if percentage_profit>0:
        if self.balance >= self.max_balance: self.max_balance = self.balance
        self.good_trades_count += 1
        if self.max_profit==0 or percentage_profit>self.max_profit:
          self.max_profit = percentage_profit
      ### LOSS
      elif percentage_profit<0:
        if self.balance <= self.min_balance: self.min_balance = self.balance
        self.bad_trades_count += 1
        if self.max_drawdown==0 or percentage_profit<self.max_drawdown:
          self.max_drawdown = percentage_profit
        if SL:
          self.SL_losses += (self.balance-self.prev_bal)
      if self.good_trades_count>0 and self.bad_trades_count>0:
        self.trades_PNL_ratio.append(self.good_trades_count/self.bad_trades_count)
      self.position_size = (self.balance*self.postition_ratio)
      self.qty = 0
      self.in_position = 0
      self.in_position_counter = 0
      self.stop_loss_price = 0