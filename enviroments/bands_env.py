from warnings import warn

from gym import spaces, Env
from numpy import array, float64, inf

from definitions import ADDITIONAL_DATA_BY_OHLCV_MA
from utils.ta_tools import get_MA_band_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


# TODO: Check gym standards for action and observation spaces
########################################################################################################################
# EXECUTING ENVIRONMENTS
class _BandsExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, take_profit=None, save_ratio=None,
              enter_at=1.0, close_at=1.0,
              atr_multi=1.0, atr_period=1,
              ma_type=0, ma_period=1, **kwargs):
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        _max_period = max(self.ma_period * ADDITIONAL_DATA_BY_OHLCV_MA[ma_type], self.atr_period)
        _ret = super().reset(*args, offset=_max_period, stop_loss=stop_loss, take_profit=take_profit,
                             save_ratio=save_ratio,
                             enter_at=enter_at, close_at=close_at, **kwargs)
        # Calculate only the data length necessary, with additional length caused by indicator periods
        if self.start_step > _max_period:
            prev_values = self.start_step - _max_period
        else:
            prev_values = 0
            warn(
                f'Previous data required for consistent MAs calculation is larger than previous values existing in df. ({_max_period} vs {self.start_step})')
        self.signals = get_MA_band_signal(self.df[prev_values:self.end_step, 1:6].astype(float),
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


class _BandsExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None, take_profit=None, save_ratio=None,
              long_enter_at=1.0, long_close_at=1.0, short_enter_at=1.0, short_close_at=1.0,
              atr_multi=1.0, atr_period=1,
              ma_type=0, ma_period=1, leverage=5, **kwargs):
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        _max_period = max(self.ma_period * ADDITIONAL_DATA_BY_OHLCV_MA[ma_type], self.atr_period)
        _ret = super().reset(*args, offset=_max_period, position_ratio=position_ratio, save_ratio=save_ratio,
                             stop_loss=stop_loss, take_profit=take_profit,
                             long_enter_at=long_enter_at, long_close_at=long_close_at,
                             short_enter_at=short_enter_at, short_close_at=short_close_at,
                             leverage=leverage, **kwargs)
        # Calculate only the data length necessary, with additional length caused by indicator periods
        if self.start_step > _max_period:
            prev_values = self.start_step - _max_period
        else:
            prev_values = 0
            warn(
                f'Previous data required for consistent MAs calculation is larger than previous values existing in df. ({_max_period} vs {self.start_step})')
        # print(f'self.df[prev_values:self.end_step, 1:6] {self.df[prev_values:self.end_step, 1:6].dtype}')
        self.signals = get_MA_band_signal(self.df[prev_values:self.end_step, 1:6].astype(float),
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


########################################################################################################################
# OPTIMIZE ENVIRONMENTS
class BandsOptimizeSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _BandsExecuteSpotEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.0001, 0.0001, 0.001, 0.001, 0.001, 2, 0, 2]
        action_upper = [0.0500, 1.0000, 1.000, 1.000, 15.0, 10_000, 37, 10_000]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(stop_loss=action[0], take_profit=action[1], enter_at=action[2], close_at=action[3],
                   atr_multi=action[4], atr_period=int(action[5]),
                   ma_type=int(action[6]), ma_period=int(action[7]))
        return self.exec_env()


class BandsOptimizeFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _BandsExecuteFuturesEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.01, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 2, 0, 2, 1]
        action_upper = [1.00, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 15.0, 10_000, 37, 10_000, 125]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1], take_profit=action[2],
                   long_enter_at=action[3], long_close_at=action[4],
                   short_enter_at=action[5], short_close_at=action[6],
                   atr_multi=action[7], atr_period=int(action[8]),
                   ma_type=int(action[9]), ma_period=int(action[10]), leverage=int(action[11]))
        return self.exec_env()


########################################################################################################################
# OPTIMIZE ENVIRONMENTS WITH SAVING BALANCE PARAMETER
class BandsOptimizeSavingSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _BandsExecuteSpotEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.000, 0.0001, 0.0001, 0.001, 0.001, 0.001, 2, 0, 2]
        action_upper = [1.000, 0.0500, 1.0000, 1.000, 1.000, 15.0, 10_000, 37, 10_000]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(save_ratio=action[0], stop_loss=action[1], take_profit=action[2],
                   enter_at=action[3], close_at=action[4],
                   atr_multi=action[5], atr_period=int(action[6]),
                   ma_type=int(action[7]), ma_period=int(action[8]))
        return self.exec_env()


class BandsOptimizeSavingFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _BandsExecuteFuturesEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.01, 0.000, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 2, 0, 2, 1]
        action_upper = [1.00, 1.000, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 15.0, 10_000, 37, 10_000, 125]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(position_ratio=action[0], save_ratio=action[1],
                   atr_period=int(action[2]), ma_type=int(action[3]),
                   ma_period=int(action[4]), leverage=int(action[5]),
                   stop_loss=action[6], take_profit=action[7],
                   long_enter_at=action[8], long_close_at=action[9],
                   short_enter_at=action[10], short_close_at=action[11],
                   atr_multi=action[12])
        return self.exec_env()
