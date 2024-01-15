from gym import spaces, Env
from numpy import array, float64, inf

from utils.ta_tools import get_MA_band_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


class BandsExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              atr_multi=1.0, atr_period=1,
              ma_type=0, ma_period=1, **kwargs):
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
        action_lower = [0.0001, 0.001, 0.001, 0.001, 2, 0, 2]
        action_upper = [0.0500, 1.000, 1.000, 15.0, 10_000, 37, 10_000]
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
                   atr_multi=action[3], atr_period=int(action[4]),
                   ma_type=int(action[5]), ma_period=int(action[6]))
        return self.exec_env()


class BandsExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None,
              long_enter_at=1.0, long_close_at=1.0, short_enter_at=1.0, short_close_at=1.0,
              atr_multi=1.0, atr_period=1,
              ma_type=0, ma_period=1, leverage=5, **kwargs):
        _ret = super().reset(*args, position_ratio=position_ratio, stop_loss=stop_loss,
                             long_enter_at=long_enter_at, long_close_at=long_close_at,
                             short_enter_at=short_enter_at, short_close_at=short_close_at,
                             leverage=leverage, **kwargs)
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
        action_lower = [0.01, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 2, 0, 2, 1]
        action_upper = [1.00, 0.0500, 1.000, 1.000, 1.000, 1.000, 15.0, 10_000, 37, 10_000, 125]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, stop_loss=None,
              long_enter_at=1.0, long_close_at=1.0,
              short_enter_at=1.0, short_close_at=1.0, leverage=1,
              ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.0):
        return self.exec_env.reset(position_ratio=position_ratio, stop_loss=stop_loss,
                                   long_enter_at=long_enter_at, long_close_at=long_close_at,
                                   short_enter_at=short_enter_at, short_close_at=short_close_at,
                                   atr_multi=atr_multi, atr_period=atr_period,
                                   ma_type=ma_type, ma_period=ma_period,
                                   leverage=leverage)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1],
                   long_enter_at=action[2], long_close_at=action[3],
                   short_enter_at=action[4], short_close_at=action[5],
                   atr_multi=action[6], atr_period=int(action[7]),
                   ma_type=int(action[8]), ma_period=int(action[9]), leverage=int(action[10]))
        return self.exec_env()
