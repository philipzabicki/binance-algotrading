from gym import spaces, Env
from numpy import array, float64, inf

from utils.ta_tools import get_MA_band_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


class BandsExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=1, atr_period=1, atr_multi=1.0, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, enter_at=enter_at, close_at=close_at, **kwargs)
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        _max_period = max(self.ma_period, self.atr_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        self.signals = get_MA_band_signal(self.df[prev_values:self.end_step, :5],
                                          self.ma_type, self.ma_period,
                                          self.atr_period, self.atr_multi)[self.start_step - prev_values:]
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' ma_type={self.ma_type}, ma_period={self.ma_period}')
            print(f' atr_period={self.atr_period}, atr_multi={self.atr_multi}')
        if self.balance >= 1_000_000:
            self.verbose = False


class BandsStratSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = BandsExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 0.001]
        action_upper = [0.0500, 1.000, 1.000, 37, 1_000, 1_000, 15.0]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.0):
        return self.exec_env.reset(stop_loss=stop_loss, enter_at=enter_at, close_at=close_at,
                                   ma_type=ma_type, ma_period=ma_period,
                                   atr_period=atr_period, atr_multi=atr_multi)

    def step(self, action):
        self.reset(stop_loss=action[0], enter_at=action[1], close_at=action[2],
                   ma_type=int(action[3]), ma_period=int(action[4]),
                   atr_period=int(action[5]), atr_multi=action[6])
        return self.exec_env()


class BandsExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None,
              leverage=5, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=1, atr_period=1, atr_multi=1.0, **kwargs):
        _ret = super().reset(*args, position_ratio=position_ratio,
                             leverage=leverage, stop_loss=stop_loss,
                             enter_at=enter_at, close_at=close_at, **kwargs)
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        _max_period = max(self.ma_period, self.atr_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        self.signals = get_MA_band_signal(self.df[prev_values:self.end_step, :5],
                                          self.ma_type, self.ma_period,
                                          self.atr_period, self.atr_multi)[self.start_step - prev_values:]
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' ma_type={self.ma_type}, ma_period={self.ma_period}')
            print(f' atr_period={self.atr_period}, atr_multi={self.atr_multi}')
        if self.balance >= 1_000_000:
            self.verbose = False


class BandsStratFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = BandsExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 0.001]
        action_upper = [0.0500, 1.000, 1.000, 37, 1_000, 1_000, 15.0]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.0):
        return self.exec_env.reset(stop_loss=stop_loss, enter_at=enter_at, close_at=close_at,
                                   ma_type=ma_type, ma_period=ma_period,
                                   atr_period=atr_period, atr_multi=atr_multi)

    def step(self, action):
        self.reset(stop_loss=action[0], enter_at=action[1], close_at=action[2],
                   ma_type=int(action[3]), ma_period=int(action[4]),
                   atr_period=int(action[5]), atr_multi=action[6])
        return self.exec_env()
