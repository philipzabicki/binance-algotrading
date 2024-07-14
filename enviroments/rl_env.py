from warnings import warn

import numpy as np
from gym import spaces
# from gymnasium import spaces
from numpy import array, inf

from .base import SpotBacktest, FuturesBacktest


class SpotRL(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Observation space #
        # other_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 10
        # As default, don't use ohlcv values from dataframe as features/obs space
        if self.exclude_cols_left < 5:
            warn(
                f'OHLCV values are not excluded from features/observation space (exclude_cols_left={self.exclude_cols_left})')
        obs_space_dims = len(self.df[0, self.exclude_cols_left:]) + other_obs_count
        obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        print(f'    observation_space {self.observation_space}')

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        trade_data = [self.qty, self.balance, self.position_size,
                      self.in_position, self.in_position_counter, self.pnl,
                      0, 0,
                      self.profit_hold_counter, self.loss_hold_counter]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        # return np.hstack((first_obs, trade_data)), self.info
        return np.hstack((first_obs, trade_data))

    # Get the data points for the given current_step
    def _next_observation(self):
        try:
            self.current_step += 1
            df_features = next(self.obs)
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            df_features = self.df[self.current_step, self.exclude_cols_left:]
        dist_to_sl = self.df[self.current_step, 3] - self.stop_loss_price if self.stop_loss is not None else 0
        dist_to_tp = self.df[self.current_step, 3] - self.take_profit_price if self.take_profit is not None else 0
        trade_data = [self.qty, self.balance, self.position_size,
                      self.in_position, self.in_position_counter, self.pnl,
                      dist_to_sl, dist_to_tp,
                      self.profit_hold_counter, self.loss_hold_counter]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        return np.hstack((df_features, trade_data))

    def _calculate_reward(self):
        # Position closed/sold #
        if self.position_closed:
            last_pnl = self.PLs_and_ratios[-1][0]
            if last_pnl > 0:
                self.reward = 10 * last_pnl * (self.good_trades_count / self.bad_trades_count)
            elif last_pnl < 0:
                self.reward = 10 * last_pnl * (self.bad_trades_count / self.good_trades_count)
            self.position_closed = 0
        # In Position #
        elif self.in_position:
            self.reward = self.pnl
            # self.reward = 0
        else:
            self.reward = 0
        # print(f'reward: {self.reward}')
        return self.reward

    def step(self, action):
        obs, _, _, _, _ = super().step(action)
        # return obs, self._calculate_reward(), self.done, False, self.info
        return obs, self._calculate_reward(), self.done, self.done, self.info

    def render(self, visualize=False, *args, **kwargs):
        super().render(*args, indicator_or_reward=self.reward, visualize=visualize, **kwargs)


class FuturesRL(FuturesBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Observation space #
        # other_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 11
        if self.exclude_cols_left < 5:
            warn(
                f'ohlcv values are not excluded from features/observation space (exclude_cols_left={self.exclude_cols_left})')
        obs_space_dims = len(self.df[0, self.exclude_cols_left:]) + other_obs_count
        obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        print(f'    observation_space {self.observation_space}')

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        trade_data = [self.qty, self.balance, self.margin,
                      self.in_position, self.in_position_counter, self.pnl,
                      0, 0, 0,
                      self.profit_hold_counter, self.loss_hold_counter]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        # return np.hstack((first_obs, trade_data)), self.info
        return np.hstack((first_obs, trade_data))

    # Get the data points for the given current_step
    def _next_observation(self):
        try:
            self.current_step += 1
            df_features = next(self.obs)
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            df_features = self.df[self.current_step, self.exclude_cols_left:]
        dist_to_liq = self.df_mark[self.current_step, 3] - self.enter_price if self.in_position else 0
        dist_to_sl = self.df[self.current_step, 3] - self.stop_loss_price if self.stop_loss is not None else 0
        dist_to_tp = self.df[self.current_step, 3] - self.take_profit_price if self.take_profit is not None else 0
        trade_data = [self.qty, self.balance, self.margin,
                      self.in_position, self.in_position_counter, self.pnl,
                      dist_to_sl, dist_to_tp, dist_to_liq,
                      self.profit_hold_counter, self.loss_hold_counter]
        if np.isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        return np.hstack((df_features, trade_data))

    def _calculate_reward(self):
        # Position closed/sold #
        if self.position_closed:
            last_pnl = self.PLs_and_ratios[-1][0]
            if last_pnl > 0:
                self.reward = last_pnl * (self.good_trades_count / self.bad_trades_count)
            elif last_pnl < 0:
                self.reward = last_pnl * (self.bad_trades_count / self.good_trades_count)
            self.position_closed = 0
        # In Position #
        elif self.in_position:
            # self.reward = self.pnl
            self.reward = 0
        else:
            self.reward = 0
        # print(f'reward: {self.reward}')
        return self.reward

    def step(self, action):
        obs, _, _, _, _ = super().step(action)
        # return obs, self._calculate_reward(), self.done, False, self.info
        return obs, self._calculate_reward(), self.done, self.info

    def render(self, visualize=False, *args, **kwargs):
        super().render(*args, indicator_or_reward=self.reward, visualize=visualize, **kwargs)
