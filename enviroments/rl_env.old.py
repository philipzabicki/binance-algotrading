# from random import normalvariate

import numpy as np
# from gym import spaces, Env
from gymnasium import spaces
from numpy import array, inf

from .base import SpotBacktest


class SpotRL(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Observation space #
        # none_df_obs_count are observations like current PnL, account balance, asset quantity etc. #
        exclude_cols_left = kwargs['exclude_cols_left'] if 'exclude_cols_left' in kwargs else 0
        other_obs_count = 12
        obs_space_dims = len(self.df[0, exclude_cols_left:]) + other_obs_count
        obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        print(f'    observation_space {self.observation_space}')

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        trade_data = [self.qty, self.in_position, self.in_position_counter,
                      self.position_closed, self.episode_orders, self.pnl,
                      self.profit_hold_counter, self.loss_hold_counter,
                      self.good_trades_count, self.bad_trades_count,
                      self.max_drawdown, self.max_profit]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        return np.hstack((first_obs, trade_data)), self.info

    # Get the data points for the given current_step
    def _next_observation(self):
        trade_data = [self.qty, self.in_position, self.in_position_counter,
                      self.position_closed, self.episode_orders, self.pnl,
                      self.profit_hold_counter, self.loss_hold_counter,
                      self.good_trades_count, self.bad_trades_count,
                      self.max_drawdown, self.max_profit]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
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
                self.reward = 100 * last_pnl * (self.good_trades_count / self.bad_trades_count)
            elif last_pnl < 0:
                self.reward = 100 * last_pnl * (self.bad_trades_count / self.good_trades_count)
            self.position_closed = 0
        # In Position #
        elif self.in_position:
            self.reward = self.pnl
            # self.reward = 0
        else:
            self.reward = 0
        # print(f'reward: {self.reward}')
        return self.reward

    def _buy(self, price):
        super()._buy(price)
        # We reset this flag not until _calculate_reward()
        self.position_closed = 1

    def step(self, action):
        obs, _, _, _, _ = super().step(action)
        return obs, self._calculate_reward(), self.done, False, self.info

    def render(self, visualize=False, *args, **kwargs):
        super().render(indicator_or_reward=self.reward, visualize=visualize)

# class FuturesRL(SpotRL):
# raise NotImplementedError
# def step(self, action):
#     close = self.df[self.current_step, 3]
#     current_close = random.uniform(round(close * (1 + self.price_slippage), 2),
#                                    round(close * (1 - self.price_slippage), 2))
#     if self.current_step == self.end_step:
#         if self.in_position: self._sell(current_close)
#         return self._finish_episode()
#     self.done = False
#     self.position_closed = 0
#     ########################## VISUALIZATION ###############################
#     if self.visualize:
#         Date = self.dates_df[self.current_step]
#         High = self.df[self.current_step, 1]
#         Low = self.df[self.current_step, 2]
#     ########################################################################
#     ## SELL OR PASS ##
#     if self.in_position:
#         self._get_pnl(current_close, update=True)
#         self.in_position_counter += 1
#         ## PASS ##
#         if action == 0:
#             pass
#         ## SELL ##
#         elif action == 2 and self.qty > 0:
#             self._sell(current_close)
#             ########################## VISUALIZATION ###############################
#             if self.visualize:  self.trades.append(
#                 {'Date': Date, 'High': High, 'Low': Low, 'total': self.qty, 'type': "close_long"})
#             ########################################################################
#             # If balance is not enough to pay 3x fee besides buys size
#             if self.balance * (1 - 3 * self.fee) <= (current_close * self.coin_step):
#                 print('Episode ended: balance(minus fee) below minimal coin step - unable to buy')
#                 return self._finish_episode()
#     ## BUY OR PASS ##
#     else:
#         ## PASS ##
#         if action == 0:
#             pass
#         # If balance is not enough to pay 3x fee besides buys size
#         elif self.balance * (1 - 3 * self.fee) <= (current_close * self.coin_step):
#             return self._finish_episode()
#         ## BUY ##
#         elif action == 1:
#             # print('OPENING LONG')
#             self._buy(current_close)
#             if self.visualize: self.trades.append(
#                 {'Date': Date, 'High': High, 'Low': Low, 'total': self.qty, 'type': "open_long"})
#     ## Finish episode if there were 0 orders after 7 days (24*60*7)
#     if (not self.episode_orders) and ((self.current_step - self.start_step) > 10_080):
#         print(f'(episode finished: {self.episode_orders} trades {self.current_step - self.start_step} steps)',
#               end='  ')
#         if self.in_position: self._sell(current_close)
#         return self._finish_episode()
#     else:
#         None
#     self._get_pnl(current_close, update=True)
#     self._calculate_reward()
#     info = {'action': action,
#             'reward': self.reward,
#             'step': self.current_step}
#     self.current_step += 1
#     return self._next_observation(), self.reward, self.done, info
#
# def _buy(self, price):
#     self.in_position = 1
#     self.episode_orders += 1
#     self.enter_price = price
#     # *(1-2*self.fee) Just to be sure we'll not run out of funds to pay fee for buy
#     step_adj_qty = floor((self.position_size) / (price * self.coin_step))
#     if step_adj_qty == 0:
#         self._finish_episode()
#     self.qty = step_adj_qty * self.coin_step
#     self.position_size = self.qty * price
#     self.balance -= self.position_size
#     fee = (self.position_size * self.fee)
#     self.balance -= fee
#     self.cumulative_fees += fee
#     if self.balance < 0:
#         self._finish_episode()
#
# def _sell(self, price):
#     self.position_closed = 1
#     self.balance += self.qty * price
#     if self.balance >= self.max_balance: self.max_balance = self.balance
#     if self.balance <= self.min_balance: self.min_balance = self.balance
#     fee = (price * abs(self.qty) * self.fee)
#     self.balance -= fee
#     self.cumulative_fees += fee
#     if self.pnl >= 0:
#         self.good_trades_count += 1
#     else:
#         self.bad_trades_count += 1
#     self.realized_PNLs.append(self.pnl)
#     self.position_size = (self.balance * self.postition_ratio)
#     self.qty = 0
#     self.in_position = 0
#     self.in_position_counter = 0
#
# def render(self, visualize=False, *args, **kwargs):
#     if visualize or self.visualize:
#         Date = self.dates_df[self.current_step]
#         Open = self.df[self.current_step, 0]
#         High = self.df[self.current_step, 1]
#         Low = self.df[self.current_step, 2]
#         Close = self.df[self.current_step, 3]
#         Volume = self.reward
#         # Render the environment to the screen
#         if self.in_position:
#             self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance + self.qty * Close,
#                                       self.trades)
#         else:
#             self.visualization.render(Date, Open, High, Low, Close, Volume, self.balance, self.trades)
