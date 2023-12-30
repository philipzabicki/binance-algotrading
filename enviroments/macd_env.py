from gym import spaces, Env
from numpy import array, float64, inf

from utils.ta_tools import custom_MACD, MACD_cross_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


class MACDExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, enter_at=enter_at, close_at=close_at, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        _max_period = max(self.fast_period, self.slow_period) + self.signal_period
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        # print(self.df[self.start_step:self.end_step, :5])
        macd, macd_signal = custom_MACD(self.df[prev_values:self.end_step, :5],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.signals = MACD_cross_signal(macd[self.start_step - prev_values:],
                                         macd_signal[self.start_step - prev_values:])
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(
                f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')


class MACDStratSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = MACDExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 2, 2, 2, 0, 0, 0]
        action_upper = [0.0500, 1.000, 1.000, 10_000, 10_000, 10_000, 37, 37, 26]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, enter_at=1.0, close_at=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=1, slow_ma_type=1, signal_ma_type=1):
        return self.exec_env.reset(stop_loss=stop_loss, enter_at=enter_at, close_at=close_at,
                                   fast_period=fast_period, slow_period=slow_period, signal_period=signal_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type, signal_ma_type=signal_ma_type)

    def step(self, action):
        self.reset(stop_loss=action[0], enter_at=action[1], close_at=action[2],
                   fast_period=int(action[3]), slow_period=int(action[4]), signal_period=int(action[5]),
                   fast_ma_type=int(action[6]), slow_ma_type=int(action[7]), signal_ma_type=int(action[8]))
        return self.exec_env()


########################################################################################################################
# FUTURES
class MACDExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None,
              enter_at=1.0, close_at=1.0, leverage=5,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, **kwargs):
        _ret = super().reset(*args, position_ratio=position_ratio, leverage=leverage, stop_loss=stop_loss,
                             enter_at=enter_at, close_at=close_at, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        _max_period = max(self.fast_period, self.slow_period) + self.signal_period
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        # print(self.df[self.start_step:self.end_step, :5])
        macd, macd_signal = custom_MACD(self.df[prev_values:self.end_step, :5],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.signals = MACD_cross_signal(macd[self.start_step - prev_values:],
                                         macd_signal[self.start_step - prev_values:])
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(
                f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')


class MACDStratFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = MACDExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.01, 0.0001, 0.001, 0.001, 1, 2, 2, 2, 0, 0, 0]
        action_upper = [1.0, 0.0500, 1.000, 1.000, 125, 10_000, 10_000, 10_000, 37, 37, 26]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, leverage=5, stop_loss=None, enter_at=1.0, close_at=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=1, slow_ma_type=1, signal_ma_type=1):
        return self.exec_env.reset(position_ratio=position_ratio, stop_loss=stop_loss,
                                   enter_at=enter_at, close_at=close_at, leverage=leverage,
                                   fast_period=fast_period, slow_period=slow_period, signal_period=signal_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type, signal_ma_type=signal_ma_type)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1],
                   enter_at=action[2], close_at=action[3], leverage=int(action[4]),
                   fast_period=int(action[5]), slow_period=int(action[6]), signal_period=int(action[7]),
                   fast_ma_type=int(action[8]), slow_ma_type=int(action[9]), signal_ma_type=int(action[10]))
        return self.exec_env()
