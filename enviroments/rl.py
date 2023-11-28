from collections import deque
# from random import normalvariate
from math import copysign, sqrt, floor
from random import randint
from time import time
import numpy as np
# from gym import spaces, Env
from gymnasium import Env, spaces
from numpy import array, inf, mean, std, random
from utility import TradingGraph


class SpotRL(Env):
    def __init__(self, df, dates_df=None, max_steps=0, exclude_cols_left=0,
                 init_balance=1_000, position_ratio=1.0, stop_loss=None, fee=0.0002, coin_step=0.001,
                 slippage=None, render_range=120, verbose=True, visualize=False):
        self.creation_t = time()
        print(f'Environment created. Fee: {fee} Coin step: {coin_step}')
        print(f' df size: {len(df)} obs sample(last row): {df[-1, exclude_cols_left:]}')
        print(f' slippage stats: {slippage}')
        if dates_df is not None:
            self.dates_df = dates_df
        if visualize:
            print(f' Visualization started, from: {dates_df[0]}', end=' ')
            print(f'to: {dates_df[-1]}')
            self.visualize = True
            self.render_range = render_range
        else:
            self.visualize = False
            print(f'    Visualize is set to false or there was no dates df provided.')
        self.verbose = verbose
        # This implementation uses only mean values provided by arg dict (slippage) #
        # as factor for calculation of real buy and sell prices. #
        # Generation of random numbers is too expensive computational wise. #
        # self.slippage = slippage
        if slippage is not None:
            self.buy_factor = slippage['market_buy'][0]
            self.sell_factor = slippage['market_sell'][0]
            self.stop_loss_factor = slippage['SL'][0]
        else:
            self.buy_factor, self.sell_factor, self.stop_loss_factor = 1.0, 1.0, 1.0
        self.df = df
        self.total_steps = len(self.df)
        self.exclude_cols_left = exclude_cols_left
        self.coin_step = coin_step
        self.fee = fee
        self.max_steps = max_steps
        self.init_balance = init_balance
        self.init_position_size = init_balance * position_ratio
        self.position_ratio = position_ratio
        self.position_size = self.init_position_size
        self.balance = self.init_balance
        self.stop_loss = stop_loss
        # Discrete action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)
        # Observation space #
        # none_df_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 12
        obs_space_dims = len(self.df[0, exclude_cols_left:]) + other_obs_count
        obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        print(f'    observation_space {self.observation_space}')

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        self.creation_t = time()
        self.total_steps = len(self.df)
        self.done = False
        self.reward = 0
        self.output = False
        if self.visualize:
            self.trades = deque(maxlen=self.render_range)
            self.visualization = TradingGraph(self.render_range)
        self.info = {}
        self.PLs_and_ratios = []
        self.balance = self.init_balance
        self.init_position_size = self.init_balance * self.position_ratio
        self.position_size = self.init_position_size
        self.prev_bal = 0
        self.enter_price = 0
        self.stop_loss_price = 0
        self.qty = 0
        self.pnl = 0
        self.SL_losses, self.cumulative_fees = 0, 0
        self.in_position, self.position_closed, self.in_position_counter, self.episode_orders = 0, 0, 0, 1
        self.good_trades_count, self.bad_trades_count = 1, 1
        self.max_drawdown, self.max_profit, self.max_balance_bonus = 0, 0, 0
        self.loss_hold_counter, self.profit_hold_counter = 1, 1
        self.max_balance = self.min_balance = self.balance
        if self.max_steps > 0:
            self.start_step = randint(0, self.total_steps - self.max_steps)
            self.end_step = self.start_step + self.max_steps - 1
        else:
            self.start_step = 0
            self.end_step = self.total_steps - 1
        self.current_step = self.start_step
        self.obs = iter(self.df[self.start_step:self.end_step, self.exclude_cols_left:])
        trade_data = [self.qty, self.in_position, self.in_position_counter,
                      self.position_closed, self.episode_orders, self.pnl, self.profit_hold_counter,
                      self.loss_hold_counter, self.good_trades_count, self.bad_trades_count, self.max_drawdown,
                      self.max_profit]
        if np.isnan(trade_data).any():
            print(trade_data)
        return np.hstack((next(self.obs), trade_data)), self.info

    # Get the data points for the given current_step
    def _next_observation(self):
        trade_data = [self.qty, self.in_position, self.in_position_counter,
                      self.position_closed, self.episode_orders, self.pnl, self.profit_hold_counter,
                      self.loss_hold_counter, self.good_trades_count, self.bad_trades_count, self.max_drawdown,
                      self.max_profit]
        if np.isnan(trade_data).any():
            print(trade_data)
        try:
            self.current_step += 1
            # _obs = next(self.obs)
            # print(_obs)
            return np.hstack((next(self.obs), trade_data))
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            return np.hstack((self.df[self.current_step, self.exclude_cols_left:], trade_data))

    def _calculate_reward(self):
        # Position closed/sold #
        if self.position_closed:
            last_pnl = self.PLs_and_ratios[-1][0]
            if last_pnl > 0:
                self.reward = 100 * last_pnl * (self.bad_trades_count / self.good_trades_count)
            elif last_pnl < 0:
                self.reward = 100 * last_pnl * (self.good_trades_count / self.bad_trades_count)
            self.in_position_counter = 0
            self.position_closed = 0
            if self.max_balance_bonus > 0:
                # print('self.max_balance_bonus>0')
                self.reward += self.max_balance_bonus * 0.01
                self.max_balance_bonus = 0
        # In Position #
        elif self.in_position:
            # self.reward = self.pnl
            self.reward = 0
        else:
            self.reward = 0
        # print(f'reward: {self.reward}')
        return self.reward

    def _buy(self, price):
        if self.stop_loss is not None:
            self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
        # Considering random factor as in real world scenario #
        # price = self._random_factor(price, 'market_buy')
        self.enter_price = price
        price = price * self.buy_factor
        # When there is no fee, subtract 1 just to be sure balance can buy this amount #
        step_adj_qty = floor((self.position_size * (1 - 2 * self.fee)) / (price * self.coin_step))
        if step_adj_qty == 0:
            self._finish_episode()
        # self.enter_price = price
        self.in_position = 1
        self.episode_orders += 1
        self.qty = round(step_adj_qty * self.coin_step, 5)
        self.position_size = round(self.qty * price, 2)
        self.prev_bal = self.balance
        self.balance -= self.position_size
        fee = (self.position_size * self.fee)
        self.position_size -= fee
        self.cumulative_fees += fee
        # print(f'BOUGHT {self.qty} at {price}')
        if self.visualize:
            self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': "open_long", 'Reward': self.df[self.current_step, -1]})

    def _sell(self, price, sl=False):
        # print(f'SOLD {self.qty} at {price}')
        if sl:
            '''price = self._random_factor(price, 'SL')
        while price>self.enter_price:
          price = self._random_factor(price, 'SL')'''
            price = price * self.stop_loss_factor
            order_type = 'open_short'
        else:
            '''price = self._random_factor(price, 'market_sell')'''
            price = price * self.sell_factor
            order_type = 'close_long'
        self.balance += round(self.qty * price, 2)
        fee = abs(price * self.qty * self.fee)
        self.balance -= fee
        self.cumulative_fees += fee
        percentage_profit = (self.balance / self.prev_bal) - 1
        # PROFIT #
        if percentage_profit >= 0:
            if self.balance >= self.max_balance:
                self.max_balance_bonus = self.balance-self.max_balance
                self.max_balance = self.balance
            self.good_trades_count += 1
            if self.max_profit == 0 or percentage_profit > self.max_profit:
                self.max_profit = percentage_profit
        # LOSS #
        elif percentage_profit < 0:
            if self.balance <= self.min_balance: self.min_balance = self.balance
            self.bad_trades_count += 1
            if self.max_drawdown == 0 or percentage_profit < self.max_drawdown:
                self.max_drawdown = percentage_profit
            if sl:
                self.SL_losses += self.prev_bal - self.balance
        self.PLs_and_ratios.append((percentage_profit, self.good_trades_count / self.bad_trades_count))
        self.position_size = (self.balance * self.position_ratio)
        # If balance minus position_size and fee is less or eq 0 #
        if self.position_size < (price * self.coin_step):
            self._finish_episode()
        self.qty = 0
        self.in_position = 0
        # self.in_position_counter = 0
        self.stop_loss_price = 0
        self.position_closed = 1
        if self.visualize:
            self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': order_type, 'Reward': self.df[self.current_step, -1]})

    def step(self, action):
        #print(f'action {action}')
        # 30-day DCA, adding 222USD to balance
        # if (self.current_step-self.start_step)%43_200 == 0:
        # self.balance+=222
        # self.init_balance+=222
        if self.in_position:
            low, close = self.df[self.current_step, 2:4]
            # print(f'low: {low}, close: {close}, self.enter_price: {self.enter_price}')
            self.in_position_counter += 1
            self.pnl = (close / self.enter_price) - 1
            #print(f'pnl: {self.pnl}')
            if self.pnl > 0:
                self.profit_hold_counter += 1
            else:
                self.loss_hold_counter += 1
            if (self.stop_loss is not None) and (low <= self.stop_loss_price):
                self._sell(self.stop_loss_price, sl=True)
            elif action == 2 and self.qty > 0:
                self._sell(close)
        elif action == 1:
            close = self.df[self.current_step, 3]
            self._buy(close)
        elif (not self.episode_orders) and ((self.current_step - self.start_step) > 2_880):
            # self._finish_episode()
            self.reward = -1
        '''if self.current_step==self.end_step:
        if self.in_position:
          close = self.df[self.current_step, 3]
          self._sell(close)
        return self._finish_episode()'''
        '''info = {'action': action,
                'in_position': self.in_position,
                'position_size': self.position_size,
                'balance': self.balance,
                'reward': 0,
                'current_step': self.current_step,
                'end_step': self.end_step}'''
        # Older version:
        # return self._next_observation(), self.reward, self.done, self.info
        return self._next_observation(), self._calculate_reward(), self.done, False, self.info

    def render(self, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            _date = self.dates_df[self.current_step]
            _open = self.df[self.current_step, 0]
            _high = self.df[self.current_step, 1]
            _low = self.df[self.current_step, 2]
            _close = self.df[self.current_step, 3]
            # Volume = self.df[self.current_step, 4]
            _volume = self.reward
            _dohlcv = [_date, _open, _high, _low, _close, _volume]
            if self.in_position:
                self.visualization.render(_dohlcv,
                                          self.balance + (self.position_size + (self.position_size * self.pnl)),
                                          self.trades)
            else:
                self.visualization.render(_dohlcv, self.balance, self.trades)

    def _finish_episode(self):
        # print('BacktestEnv._finish_episode()')
        if self.in_position:
            self._sell(self.enter_price)
        self.done = True
        # Summary
        self.PNL_arrays = array(self.PLs_and_ratios)
        gain = self.balance - self.init_balance
        total_return = (self.balance / self.init_balance) - 1
        risk_free_return = (self.df[-1, 3] / self.df[0, 3]) - 1
        hold_ratio = self.profit_hold_counter / self.loss_hold_counter if self.loss_hold_counter > 1 and self.profit_hold_counter > 1 else 0.0
        if len(self.PNL_arrays) > 1:
            mean_pnl, stddev_pnl = mean(self.PNL_arrays[:, 0]), std(self.PNL_arrays[:, 0])
            profits = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] > 0]
            losses = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0]
            profits_mean = mean(profits) if len(profits) > 1 else 0.0
            losses_mean = mean(losses) if len(losses) > 1 else 0.0
            losses_stddev = std(losses) if len(losses) > 1 else 0.0
            PnL_trades_ratio = mean(self.PNL_arrays[:, 1])
            PnL_means_ratio = abs(profits_mean / losses_mean) if profits_mean != 0 and losses_mean != 0 else 0.0
            # slope_indicator = linear_slope_indicator(PnL_trades_ratio)
            slope_indicator = 1.000
        else:
            mean_pnl, stddev_pnl = 0.0, 0.0
            profits_mean, losses_mean, losses_stddev = 0.0, 0.0, 0.0
            PnL_trades_ratio, PnL_means_ratio = 0.0, 0.0
            slope_indicator = 1.000
        sharpe_ratio = (mean_pnl - risk_free_return) / stddev_pnl if stddev_pnl != 0 else -1
        sortino_ratio = (total_return - risk_free_return) / losses_stddev if losses_stddev != 0 else -1
        # self.reward = (copysign(abs(gain) ** 1.5, gain) * self.episode_orders * (PnL_trades_ratio ** 2) * sqrt(hold_ratio) * sqrt(PnL_means_ratio)) / self.total_steps
        # self.reward = copysign((abs(gain)**1.5)*self.PL_count_mean*sqrt(hold_ratio)*sqrt(self.PL_ratio)*sqrt(self.episode_orders), gain)/self.total_steps
        # self.reward = copysign(gain**2, gain)+(self.episode_orders/sqrt(self.total_steps))+self.PL_count_mean+sqrt(hold_ratio)+sqrt(self.PL_ratio)
        exec_time = time() - self.creation_t
        if self.verbose:
            print(
                f'Episode finished: gain:${gain:.2f}, cumulative_fees:${self.cumulative_fees:.2f}, SL_losses:${self.SL_losses:.2f}')
            print(
                f' episode_orders:{self.episode_orders:_}, trades_count(profit/loss):{self.good_trades_count - 1:_}/{self.bad_trades_count - 1:_}, ',
                end='')
            print(f'trades_avg(profit/loss):{profits_mean * 100:.2f}%/{losses_mean * 100:.2f}%, ', end='')
            print(f'max(profit/drawdown):{self.max_profit * 100:.2f}%/{self.max_drawdown * 100:.2f}%,')
            print(f' PnL_trades_ratio:{PnL_trades_ratio:.3f}, PnL_means_ratio:{PnL_means_ratio:.3f}, ', end='')
            print(f'hold_ratio:{hold_ratio:.3f}, PNL_mean:{mean_pnl * 100:.2f}%')
            print(
                f' slope_indicator:{slope_indicator:.4f}, sharpe_ratio:{sharpe_ratio:.2f}, sortino_ratio:{sortino_ratio:.2f}')
            print(f' reward:{self.reward:.8f} exec_time:{exec_time:.2f}s')

        self.info = {'gain': gain,
                     'PnL_means_ratio': PnL_means_ratio,
                     'PnL_trades_ratio': PnL_trades_ratio,
                     'hold_ratio': hold_ratio,
                     'PNL_mean': mean_pnl,
                     'slope_indicator': slope_indicator,
                     'exec_time': exec_time}


class FuturesRL(SpotRL):
    def step(self, action):
        close = self.df[self.current_step, 3]
        current_close = random.uniform(round(close * (1 + self.price_slippage), 2),
                                       round(close * (1 - self.price_slippage), 2))
        if self.current_step == self.end_step:
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
            self.in_position_counter += 1
            ## PASS ##
            if action == 0:
                pass
            ## SELL ##
            elif action == 2 and self.qty > 0:
                self._sell(current_close)
                ########################## VISUALIZATION ###############################
                if self.visualize:  self.trades.append(
                    {'Date': Date, 'High': High, 'Low': Low, 'total': self.qty, 'type': "close_long"})
                ########################################################################
                # If balance is not enough to pay 3x fee besides buys size
                if self.balance * (1 - 3 * self.fee) <= (current_close * self.coin_step):
                    print('Episode ended: balance(minus fee) below minimal coin step - unable to buy')
                    return self._finish_episode()
        ## BUY OR PASS ##
        else:
            ## PASS ##
            if action == 0:
                pass
            # If balance is not enough to pay 3x fee besides buys size
            elif self.balance * (1 - 3 * self.fee) <= (current_close * self.coin_step):
                return self._finish_episode()
            ## BUY ##
            elif action == 1:
                # print('OPENING LONG')
                self._buy(current_close)
                if self.visualize: self.trades.append(
                    {'Date': Date, 'High': High, 'Low': Low, 'total': self.qty, 'type': "open_long"})
        ## Finish episode if there was 0 orders after 7 days (24*60*7)
        if (not self.episode_orders) and ((self.current_step - self.start_step) > 10_080):
            print(f'(episode finished: {self.episode_orders} trades {self.current_step - self.start_step} steps)',
                  end='  ')
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
        step_adj_qty = floor((self.position_size) / (price * self.coin_step))
        if step_adj_qty == 0:
            self._finish_episode()
        self.qty = step_adj_qty * self.coin_step
        self.position_size = self.qty * price
        self.balance -= self.position_size
        fee = (self.position_size * self.fee)
        self.balance -= fee
        self.cumulative_fees += fee
        if self.balance < 0:
            self._finish_episode()

    def _sell(self, price):
        self.position_closed = 1
        self.balance += self.qty * price
        if self.balance >= self.max_balance: self.max_balance = self.balance
        if self.balance <= self.min_balance: self.min_balance = self.balance
        fee = (price * abs(self.qty) * self.fee)
        self.balance -= fee
        self.cumulative_fees += fee
        if self.pnl >= 0:
            self.good_trades_count += 1
        else:
            self.bad_trades_count += 1
        self.realized_PNLs.append(self.pnl)
        self.position_size = (self.balance * self.postition_ratio)
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
            if self.in_position:
                self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance + self.qty * Close,
                                          self.trades)
            else:
                self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)
