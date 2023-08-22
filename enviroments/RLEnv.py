import numpy as np
import math
import gym
import time
from gym import spaces
#import time
from collections import deque
import random
from statistics import mean, stdev

from visualize import TradingGraph

class RLEnv(gym.Env):
    def __init__(self, df, excluded_left=0, init_balance=100, postition_ratio=1.0, leverage=1, fee=0.0004, slippage=0.0001, max_steps=0, Render_range=120, visualize=False, dates_df=None):
        self.dates_df = dates_df
        self.df = df
        self.exclude_count = excluded_left
        self.df_total_steps = len(self.df)-1
        self.coin_step = 0.001
        self.price_slippage = slippage
        self.fee = fee
        self.funding_rate = 0.015 * (1/100)
        self.max_steps = max_steps
        self.init_balance = init_balance
        self.init_postition_size = init_balance*postition_ratio
        self.postition_ratio = postition_ratio
        self.position_size = self.init_postition_size
        self.balance = self.init_balance
        self.leverage = leverage
        self.Render_range = Render_range
        self.visualize = visualize
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = spaces.Discrete(3)
        obs_space_dims = len(self.df[0, self.exclude_count:])+9
        lower_bounds = np.array([-np.inf for _ in range(obs_space_dims)])
        upper_bounds = np.array([np.inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds)

    # Reset the state of the environment to an initial state
    def reset(self):
        self.done = False
        if self.visualize: 
          self.trades = deque(maxlen=self.Render_range)
          self.visualization = TradingGraph(Render_range=self.Render_range)
        self.balance = self.init_balance
        self.position_size = self.init_postition_size
        self.qty = 0
        self.cumulative_fees = 0
        self.enter_price = 0
        self.liquidations = 0
        self.in_position = 0
        self.position_closed = 0
        self.in_position_counter = 0
        self.episode_orders = 0
        self.good_trades_count = 1
        self.bad_trades_count = 1
        self.realized_PNLs = []
        self.pnl = 0
        #self.cumulative_pnl = 0
        self.max_balance = self.balance
        self.min_balance = self.balance
        self.max_balance_bonus = 0
        #self.balance_history = [self.balance, self.balance]
        self.reward = 0

        if self.max_steps > 0:
            self.start_step = random.randint(0, self.df_total_steps - self.max_steps)
            self.end_step = self.start_step + self.max_steps
        else:
            self.start_step = 0
            self.end_step = self.df_total_steps
        self.current_step = self.start_step
        return self._next_observation()

    # Get the data points for the given current_step
    def _next_observation(self):
      orders_history = np.array([self.position_size, self.balance, self.cumulative_fees, self.liquidations, self.in_position, self.in_position_counter, 
                        math.copysign(1, self.qty), self.pnl, self.df[self.current_step, 3]-self.enter_price], dtype="float32")
      obs = np.concatenate((self.df[self.current_step, self.exclude_count:], orders_history), axis=None, dtype="float32")
      #print(obs)
      return obs
    
    def _calculate_reward(self):
      ###### Postion closing ######
      #print('._calculate_reward()')
      _realized_PNLs = len(self.realized_PNLs)
      if self.position_closed and _realized_PNLs>1:
        #print('self.position_closed')
        mean_return = sum(self.realized_PNLs) / _realized_PNLs
        self.reward = (self.pnl*5)+mean_return
        if self.max_balance_bonus>0:
          #print('self.max_balance_bonus>0')
          self.reward += self.max_balance_bonus*0.01
          self.max_balance_bonus = 0
      ###### In postion ######
      elif self.in_position:
        self.reward=self.pnl
        #self.reward=0
      else:
        self.reward=0
      #print(f'self.reward {self.reward} self.pnl {self.pnl}')
      #time.sleep(1)
      return self.reward

    def _get_pnl(self, price, update=False):
      _pnl = (((price/self.enter_price)-1)*self.leverage)*math.copysign(1, self.qty)
      if update: 
        self.pnl = _pnl
        #self.pnl_list.append(self.pnl)
      elif not self.in_position:
        self.pnl = 0
      '''if self.cumulative_pnl!=0 and len(self.pnl_list)>1:
        self.cumulative_pnl+=(self.pnl-self.pnl_list[-2])
      else:
        self.cumulative_pnl+=self.pnl'''
      return _pnl

    def _finish_episode(self):
      self.done = True
      self._calculate_reward()
      profit = ((self.balance/self.init_balance)-1)*100
      gain = self.balance-self.init_balance
      if self.episode_orders>0:
        profit_mean = mean([pnl if pnl>0 else 0 for pnl in self.realized_PNLs])
        loss_mean = mean([pnl if pnl<0 else 0 for pnl in self.realized_PNLs])
        print(f' Profit: {profit:.2f}% Zysk: ${gain:.2f} Koncowy: ${self.balance:.2f} Max: ${self.max_balance:.2f}', end='  ')
        zs_ratio = self.good_trades_count/self.bad_trades_count
        print(f'z/s: {zs_ratio:.3f} zyskownych:{self.good_trades_count:} ({profit_mean*100:.2f}%) stratnych:{self.bad_trades_count} ({loss_mean*100:.2f}%) ogolem:{self.episode_orders}', end=' ')
        print(f'likwidacji:{self.liquidations} prowizje ${self.cumulative_fees:.2f}')
        info = {'performance': profit,
              'profit': gain,
              'final': self.balance,
              'max': self.max_balance,
              'profit_loss_ratio': zs_ratio,
              'profit_count': self.good_trades_count,
              'profit_avg': profit_mean*100,
              'loss_count': self.bad_trades_count,
              'loss_avg': loss_mean*100,
              'orders_count': self.episode_orders,
              'liquidations': self.liquidations,
              'fees': self.cumulative_fees}
      else:
        print(f'likwidacji:{self.liquidations} pozycji:{self.episode_orders}')
        info = {'performance': profit,
              'profit': gain,
              'final': self.balance,
              'max': self.max_balance,
              'profit_loss_ratio': 0,
              'profit_count': 0,
              'profit_avg': 0,
              'loss_count': 0,
              'loss_avg': 0,
              'orders_count': self.episode_orders,
              'liquidations': self.liquidations,
              'fees': self.cumulative_fees}
      return self._next_observation(), self.reward, self.done, info

    def _open_position(self, side, price):
      #print(f'OPENING {side} at price {price}')
      self.in_position = 1
      self.episode_orders += 1
      self.enter_price = price
      adj_qty = math.floor(self.position_size*self.leverage/(price*self.coin_step))
      if adj_qty==0: adj_qty=1
      self.position_size = (adj_qty*self.coin_step*price)/self.leverage
      #print(f'  (position size {self.position_size} qty {adj_qty})')
      if self.position_size>self.balance: return self._finish_episode()
      self.balance -= self.position_size
      fee = (self.position_size*self.fee*self.leverage)
      #print(f'  (fee {fee:.8f})')
      self.balance -= fee
      self.cumulative_fees += fee
      #print(f' balance(minus pos and fee) {self.balance}')
      if side=='long':
        self.qty = (self.position_size*self.leverage)/price
      elif side=='short':
        self.qty= -1*(self.position_size*self.leverage)/price
      

    def _close_position(self, price, liquidated=False):
      #print(f'CLOSING position at price {price}')
      self.position_closed = 1
      _position_size = self.position_size
      if liquidated:
        self.position_size*=(-1)
      else:
        self.position_size += (self.position_size*self.pnl)
      #print(f' realized PNL {_position_size*self.pnl}')
      self.balance += self.position_size
      if self.balance >= self.max_balance:
        self.max_balance_bonus = self.max_balance - self.balance
        self.max_balance = self.balance
      #print(f' new balance {self.balance}')
      fee = (price*abs(self.qty))*(self.fee*self.leverage)
      #print(f'  (fee {fee:.8f})')
      self.balance -= fee
      self.cumulative_fees += fee
      if self.pnl>0:
        self.good_trades_count += 1
      elif self.pnl<0:
        self.bad_trades_count += 1
      self.realized_PNLs.append((self.position_size/_position_size)-1)
      self.position_size = (self.balance*self.postition_ratio)
      self.qty = 0
      self.in_position = 0
      self.in_position_counter = 0
    
    # Execute one time step within the environment
    def step(self, action):
        close = self.df[self.current_step, 3]
        current_close = random.uniform(round(close*(1+self.price_slippage), 2), round(close*(1-self.price_slippage), 2))
        if self.current_step==self.end_step:
          if self.in_position: self._close_position(current_close)
          return self._finish_episode()
        self.done = False
        self.position_closed = 0
        ########################## VISUALIZATION ###############################
        if self.visualize:
          Date = self.dates_df[self.current_step]
          High = self.df[self.current_step, 1]
          Low = self.df[self.current_step, 2]
        ########################################################################
        if self.in_position:
          self._get_pnl(current_close, update=True)
          self.in_position_counter+=1
          # I assume that funding rate always adds to lose. It's calculated every 8 hours.
          if self.in_position_counter%480==0:
            self.position_size -= (self.position_size*self.funding_rate)
          if (self.qty>0 and self._get_pnl(self.df[self.current_step, 2])<=-0.9) or (self.qty<0 and self._get_pnl(self.df[self.current_step, 1])<=-0.9):
            self.liquidations += 1
            self._close_position(current_close, liquidated=True)
          else:
            ##CLOSING POSTIONS OR PASS##
            if action == 0:
              #print('IN POSITION PASS')
              pass
            # Close SHORT or LONG qty={-1,0,1}
            elif (action==1 and self.qty<0) or (action==2 and self.qty>0):
              #print('CLOSING POSITION')
              self._close_position(current_close)
              ########################## VISUALIZATION ###############################
              if self.visualize and action==1:  self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "close_short"})
              elif self.visualize and action==2:  self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "close_long"})
              ########################################################################
              # If balance is not enough to pay 3x fee both leverage adjusted 
              fee_adj_bal = self.balance*(1-3*(self.fee*self.leverage))
              one_q_price = ((current_close*self.coin_step)/self.leverage)
              if fee_adj_bal<=one_q_price:
                print(f'fee adj Balance ${fee_adj_bal:.3f} not enough to pay for one quqntity ${one_q_price:.3f}')
                return self._finish_episode()
        ##OPENING POSTIONS OR PASS##
        else:
          if action == 0:
            #print('NO POSITION PASS')
            pass
          # If balance is not enough to pay 3x fee both leverage adjusted 
          fee_adj_bal = self.balance*(1-3*(self.fee*self.leverage))
          one_q_price = ((current_close*self.coin_step)/self.leverage)
          if fee_adj_bal<=one_q_price:
            print(f'fee adj Balance ${fee_adj_bal:.3f} not enough to pay for one quqntity ${one_q_price:.3f}')
            return self._finish_episode()
          elif action == 1:
            #print('OPENING LONG')
            self._open_position('long', current_close)
            if self.visualize: self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "open_long"})
          # OPEN SHORT
          elif action == 2:
            #print('OPENING SHORT')
            self._open_position('short', current_close)
            if self.visualize: self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "open_short"})
        ## Finish episode if there was 0 orders after 7 days (24*60*7)
        if (not self.episode_orders) and ((self.current_step-self.start_step)>10_080):
          print(f'(episode finished: {self.episode_orders} trades {self.current_step-self.start_step} steps)', end='  ')
          if self.in_position: self._close_position(current_close)
          return self._finish_episode()
        else:
          None
        self._calculate_reward()
        info = {'action': action,
                'reward': self.reward,
                'step': self.current_step}
        self.current_step += 1
        return self._next_observation(), self.reward, self.done, info
        
    # Render environment
    def render(self, visualize=False, *args, **kwargs):
      if visualize or self.visualize:
        Date = self.dates_df[self.current_step]
        Open = self.df[self.current_step, 0]
        High = self.df[self.current_step, 1]
        Low = self.df[self.current_step, 2]
        Close = self.df[self.current_step, 3]
        Volume = self.reward
        # Render the environment to the screen
        if self.in_position: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance+(self.position_size+(self.position_size*self.pnl)), self.trades)
        else: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)

class RLEnvSpot(RLEnv):
  def step(self, action):
        close = self.df[self.current_step, 3]
        current_close = random.uniform(round(close*(1+self.price_slippage), 2), round(close*(1-self.price_slippage), 2))
        if self.current_step==self.end_step:
          if self.in_position: self._sell(current_close)
          return self._finish_episode()
        self.done = False
        self.position_closed = 0
        ########################## VISUALIZATION ###############################
        if self.visualize:
          Date = self.dates_df[self.current_step]
          High = self.df[self.current_step, 1]
          Low = self.df[self.current_step, 2]
        ########################################################################
        ## SELL OR PASS ##
        if self.in_position:
          self._get_pnl(current_close, update=True)
          self.in_position_counter+=1
          ## PASS ##
          if action == 0:
              pass
          ## SELL ##
          elif action==2 and self.qty>0:
              self._sell(current_close)
              ########################## VISUALIZATION ###############################
              if self.visualize:  self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "close_long"})
              ########################################################################
              # If balance is not enough to pay 3x fee besides buys size
              if self.balance*(1-3*self.fee)<=(current_close*self.coin_step):
                print('Episode ended: balance(minus fee) below minimal coin step - unable to buy')
                return self._finish_episode()
        ## BUY OR PASS ##
        else:
          ## PASS ##
          if action == 0:
            pass
          # If balance is not enough to pay 3x fee besides buys size
          elif self.balance*(1-3*self.fee)<=(current_close*self.coin_step):
            return self._finish_episode()
          ## BUY ##
          elif action == 1:
            #print('OPENING LONG')
            self._buy(current_close)
            if self.visualize: self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.qty, 'type': "open_long"})
        ## Finish episode if there was 0 orders after 7 days (24*60*7)
        if (not self.episode_orders) and ((self.current_step-self.start_step)>10_080):
          print(f'(episode finished: {self.episode_orders} trades {self.current_step-self.start_step} steps)', end='  ')
          if self.in_position: self._sell(current_close)
          return self._finish_episode()
        else:
          None
        self._get_pnl(current_close, update=True)
        self._calculate_reward()
        info = {'action': action,
                'reward': self.reward,
                'step': self.current_step}
        self.current_step += 1
        return self._next_observation(), self.reward, self.done, info

  def _buy(self, price):
      self.in_position = 1
      self.episode_orders += 1
      self.enter_price = price
      # *(1-2*self.fee) Just to be sure we'll not run out of funds to pay fee for buy
      step_adj_qty = math.floor((self.position_size)/(price*self.coin_step))
      if step_adj_qty==0:
        self._finish_episode()
      self.qty = step_adj_qty*self.coin_step
      self.position_size = self.qty*price
      self.balance -= self.position_size
      fee = (self.position_size*self.fee)
      self.balance -= fee
      self.cumulative_fees += fee
      if self.balance<0:
        self._finish_episode()

  def _sell(self, price):
      self.position_closed = 1
      self.balance += self.qty*price
      if self.balance >= self.max_balance: self.max_balance = self.balance
      if self.balance <= self.min_balance: self.min_balance = self.balance
      fee = (price*abs(self.qty)*self.fee)
      self.balance -= fee
      self.cumulative_fees += fee
      if self.pnl>=0: self.good_trades_count += 1
      else: self.bad_trades_count += 1
      self.realized_PNLs.append(self.pnl)
      self.position_size = (self.balance*self.postition_ratio)
      self.qty = 0
      self.in_position = 0
      self.in_position_counter = 0
  
  def render(self, visualize=False, *args, **kwargs):
      if visualize or self.visualize:
        Date = self.dates_df[self.current_step]
        Open = self.df[self.current_step, 0]
        High = self.df[self.current_step, 1]
        Low = self.df[self.current_step, 2]
        Close = self.df[self.current_step, 3]
        Volume = self.reward
        # Render the environment to the screen
        if self.in_position: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance+self.qty*Close, self.trades)
        else: self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)